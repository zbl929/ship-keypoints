dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='lefttop', id=0, color=[0, 153, 255], type='upper', swap='righttop'),
        1:
        dict(
            name='leftdown',
            id=1,
            color=[151, 0, 255],
            type='upper',
            swap='rightdown'),
        2:
        dict(
            name='rightdown',
            id=2,
            color=[151, 153, 0],
            type='upper',
            swap='leftdown'),
        3:
        dict(
            name='righttop',
            id=3,
            color=[151, 153, 255],
            type='upper',
            swap='lefttop'),

    },
    skeleton_info={
        0:
        dict(link=('lefttop', 'leftdown'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('leftdown', 'rightdown'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('rightdown', 'righttop'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('righttop', 'lefttop'), id=3, color=[255, 128, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.0
    ],
    sigmas=[
         0.087, 0.087, 0.089, 0.089
    ])
