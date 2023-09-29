dataset_info = dict(
    dataset_name='speedplus',
    paper_info=dict(
        author='Park, Tae Ha and Maertens, Marcus and '
        'Lecuyer, Gurvan and Izzo, Dario and DAmico, Simone',
        title='Next Generation Spacecraft Pose Estimation Dataset (SPEED+)',
        container='Zenodo',
        year='2021',
        homepage='https://zenodo.org/record/5588480'
    ),
    keypoint_info={
        0:
        dict(name='kpt_0',
             id=0,
             color=[51, 153, 255],
             type='upper',
             swap=''),
        1:
        dict(
            name='kpt_1',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='kpt_2',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='kpt_3',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='kpt_4',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='kpt_5',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='kpt_6',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        7:
        dict(
            name='kpt_7',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        8:
        dict(
            name='kpt_8',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        9:
        dict(
            name='kpt_9',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        10:
        dict(
            name='kpt_10',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('kpt_1', 'kpt_2'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('kpt_2', 'kpt_3'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('kpt_3', 'kpt_4'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('kpt_4', 'kpt_5'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('kpt_5', 'kpt_6'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('kpt_6', 'kpt_7'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('kpt_7', 'kpt_8'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('kpt_8', 'kpt_9'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('kpt_9', 'kpt_10'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('kpt_10', 'kpt_0'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('kpt_0', 'kpt_1'), id=10, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025
    ])