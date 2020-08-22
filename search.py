import os
import random
import time

SAMPLE_NUMBER = 100
extra_tag = 'seg-'

for index in range(SAMPLE_NUMBER):
    input_channel = [str(random.randint(1, 4) * 16) for _ in range(2)]
    block_1 = [str(random.randint(1, 3)),  # number of blocks
               '1'] + \
              [str(random.randint(1, 4)) for _ in range(1)] + \
              [str(random.randint(1, 8) * 16) for _ in range(1)]  # number of channels

    block_2 = [str(random.randint(1, 3)), '2'] + \
              [str(random.randint(1, 4)) for _ in range(2)] + \
              [str(random.randint(1, 8) * 16) for _ in range(2)]

    block_3 = [str(random.randint(1, 3)), '3'] + \
              [str(random.randint(1, 4)) for _ in range(3)] + \
              [str(random.randint(1, 8) * 16) for _ in range(3)]

    block_4 = [str(random.randint(1, 3)), '4'] + \
              [str(random.randint(1, 4)) for _ in range(4)] + \
              [str(random.randint(1, 8) * 16) for _ in range(4)]

    last_channel = [str(random.randint(2, 8) * 16)]
    # head = [str(random.randint(1, 8) * 32) for _ in range(4)]

    params = [input_channel, block_1, block_2, block_3, block_4, last_channel]
    params = ['_'.join(item) for item in params]
    params = extra_tag + '-'.join(params)

    command = 'PORT=9099 bash ./tools/dist_train.sh ' \
              'configs/cityscapes_hrnet_512x1024_40k.py 8 --net_params {} --work_dir ./output/{}'.format(params, params)


    print(command)
    os.system(command)

