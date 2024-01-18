# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
import warnings
from argparse import ArgumentParser
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_list_of
from typing import List
from mmpose.structures.bbox.transforms import get_warp_matrix
from mmpose.structures.pose_data_sample import PoseDataSample

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from functools import partial
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples, split_instances

from mmpose.utils import adapt_mmdet_pipeline
from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


# def process_one_image(args,
#                       img,
#                       detector,
#                       pose_estimator,
#                       visualizer=None,
#                       show_interval=0):
#     """Visualize predicted keypoints (and heatmaps) of one image."""
#
#     # predict bbox
#     det_result = inference_detector(detector, img)
#     pred_instance = det_result.pred_instances.cpu().numpy()
#     print(pred_instance)
#     bboxes = np.concatenate(
#         (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
#     bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
#                                    pred_instance.scores > args.bbox_thr)]
#     bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
#     print(bboxes)
#     # predict keypoints
#     pose_results = inference_topdown(pose_estimator, img, bboxes)
#     print(pose_results)
#     data_samples = merge_data_samples(pose_results)
#
#     # show the results
#     if isinstance(img, str):
#         img = mmcv.imread(img, channel_order='rgb')
#     elif isinstance(img, np.ndarray):
#         img = mmcv.bgr2rgb(img)
#
#     if visualizer is not None:
#         visualizer.add_datasample(
#             'result',
#             img,
#             data_sample=data_samples,
#             draw_gt=False,
#             draw_heatmap=args.draw_heatmap,
#             draw_bbox=args.draw_bbox,
#             show_kpt_idx=args.show_kpt_idx,
#             skeleton_style=args.skeleton_style,
#             show=args.show,
#             wait_time=show_interval,
#             kpt_thr=args.kpt_thr)
#
#     # if there is no instance detected, return None
#     return data_samples.get('pred_instances', None)

def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError('Invalid input type, should be a list of '
                         ':obj:`PoseDataSample`')

    if len(data_samples) == 0:
        warnings.warn('Try to merge an empty list of data samples.')
        return PoseDataSample()

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    # if 'gt_instances' in data_samples[0]:
    #     merged.gt_instances = InstanceData.cat(
    #         [d.gt_instances for d in data_samples])

    merged_instances = []
    # for i in range(len(data_samples)):
    #     merged.pred_instances
    if 'pred_instances' in data_samples[0]:
        merged.pred_instances = InstanceData.cat(
            [d.pred_instances for d in data_samples])

    track_id=[]
    if 'track_id' in data_samples[0]:
        # merged.track_id = int.cat(
        #    [d.track_id for d in data_samples])
        for i in range(len(data_samples)):
            track_id.append(data_samples[i].track_id)
            merged.set_field(track_id,'track_id')
    # merged.pred_instances
    # print('merged:',merged)
    # merged_instances = []
    # for i in range(len(data_samples)):
    #     merged_instance = dict(
    #         pred_instances=data_samples[i].pred_instances,
    #         track_id=data_samples[i].track_id,
    #     )
    #     merged_instances.append(merged_instance)
    # print('merged_instances:',merged_instances)
    # merged.pred_instances = InstanceData.cat(merged_instances)
        # if 'pred_instances' in data_samples[i]:
        #     merged_instances = []
        #     for d, nid in zip(data_samples):
        #         d.pred_instances.next_ids = np.full(len(d.pred_instances), nid)
        #         print(d.pred_instances.next_ids)
        #         merged_instances.append(d.pred_instances)
        #         print('merged_instances:',merged_instances)
        #         print('d.pred_instances:',d.pred_instances)
        #     merged.pred_instances = InstanceData.cat(merged_instances)
        #     print('merged.pred_instances:',merged.pred_instances)
    return merged
def split_instances(instances: PoseDataSample) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.pred_instances.keypoints)):
       # if 'track_id' in instances:
       #      result['track_id'] = instances.track_id[i]
       result = dict(
            # track_id=instances.pred_instances.track_id[i].tolist(),
            track_id=instances.track_id[i],
            keypoints=instances.pred_instances.keypoints[i].tolist(),
            # keypoint_scores=instances.pred_instances.keypoint_scores[i].tolist(),
        )
       if 'bboxes' in instances.pred_instances:
            result['bbox'] = instances.pred_instances.bboxes[i].tolist()

       results.append(result)

    return results

def process_one_image(args, detector, frame, frame_idx, pose_estimator,
                      pose_est_results_last, pose_est_results_list, next_id,
                       visualize_frame, visualizer):

    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # filter out the person instances with category and bbox threshold
    # e.g. 0 for person in COCO
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]

    # estimate pose results for current image
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)
    if args.use_oks_tracking:
        _track = partial(_track_by_oks)
    else:
        _track = _track_by_iou

    # convert 2d pose estimation results into the format for pose-lifting
    # such as changing the keypoint order, flipping the keypoint, etc.
    for i, data_sample in enumerate(pose_est_results):

        pred_instances = data_sample.pred_instances.cpu().numpy()
        keypoints = pred_instances.keypoints
        # calculate area and bbox
        if 'bboxes' in pred_instances:
            areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                              for bbox in pred_instances.bboxes])
            pose_est_results[i].pred_instances.set_field(areas, 'areas')
        else:
            areas, bboxes = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes.append([xmin, ymin, xmax, ymax])
            pose_est_results[i].pred_instances.areas = np.array(areas)
            pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

        # track id
        track_id, pose_est_results_last, _ = _track(data_sample,
                                                    pose_est_results_last,
                                                    args.tracking_thr)
        if track_id == -1:
            if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                track_id = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                keypoints[:, :, 1] = -10
                pose_est_results[i].pred_instances.set_field(
                    keypoints, 'keypoints')
                pose_est_results[i].pred_instances.set_field(
                    pred_instances.bboxes * 0, 'bboxes')
                pose_est_results[i].set_field(pred_instances, 'pred_instances')
                track_id = -1
        pose_est_results[i].set_field(track_id, 'track_id')


    # print('pose_est_results:',pose_est_results)

    data_samples = merge_data_samples(pose_est_results)


    # Visualization
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            visualize_frame,
            data_sample=data_samples,
            # det_data_sample=det_data_sample,
            draw_gt=False,
            # dataset_2d=pose_det_dataset_name,
            # dataset_3d=pose_lift_dataset_name,
            show=args.show,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            wait_time=args.show_interval)
    # print(pose_est_results)
    # return pose_est_results, pose_est_results_list, next_id,data_samples.get('pred_instances', None)
    return pose_est_results, pose_est_results_list, next_id,data_samples

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', default='rtmdet_s_ship.py',help='Config file for detection')
    parser.add_argument('--det_checkpoint', default=r'det_s_epoch_300.pth',help='Checkpoint file for detection')
    parser.add_argument('--pose_config', default=r'D:\CV_project\mmpose-main\tools\logs_1\rtmpose_m_ship\rtmpose-m_ship.py',help='Config file for pose')
    parser.add_argument('--pose_checkpoint',default=r'D:\CV_project\mmpose-main\tools\logs_1\rtmpose_m_ship\best_coco_AP_epoch_300.pth', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default=r"E:\船舶数据\船舶视频\16.mp4", help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default=r'D:\CV_project\mmpose-main\demo\topdown_TRACK_test',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.5,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.5,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--use-oks-tracking', default=False,action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.6, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    # print('visualizer:',visualizer)
    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)
        video_writer = None

        next_id = 0
        frame_idx = 0
        pose_est_results = []
        pose_est_results_list = []
        pred_instances_list = []

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break
            pose_est_results_last = pose_est_results
            # topdown pose estimation
            # pred_instances = process_one_image(args, frame, detector,
            #                                    pose_estimator, visualizer,
            #                                    0.001)
            (pose_est_results, pose_est_results_list,next_id,pose_results) = process_one_image(
                args=args,
                detector=detector,
                frame=frame,
                frame_idx=frame_idx,
                pose_estimator=pose_estimator,
                pose_est_results_last=pose_est_results_last,
                pose_est_results_list=pose_est_results_list,
                next_id=next_id,
                visualize_frame=mmcv.bgr2rgb(frame),
                visualizer=visualizer)

            print('pose_results:',pose_results)
            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pose_results)))
            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
