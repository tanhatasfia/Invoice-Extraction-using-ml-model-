import re
num_classes = 5
num_steps = 8000  # or 1000, depending on experiment results
num_eval_steps = 3000 #500
batch_size = 4
learning_rate_base = 0.08
warmup_learning_rate = 0.008
warmup_steps = 1000
total_steps = 20000
replicas_to_aggregate = 1
checkpoint_type = 'detection'

train_record_path = '/content/Dataset-images/train.record'
test_record_path = '/content/Dataset-images/val.record'
model_dir = '/content/training/'
labelmap_path = '/content/label_map.pbtxt'



#pipeline_config_path = '/content/frcnn_v1/pipeline.config'
#fine_tune_checkpoint = '/content/frcnn_v1/checkpoint/ckpt-0'

#pipeline_config_path = '/content/frcnn_resnet101_v1/pipeline.config'
#fine_tune_checkpoint = '/content/frcnn_resnet101_v1/checkpoint/ckpt-0'




#pipeline_config_path = '/content/efficient_d2.config'
#fine_tune_checkpoint = '/content/efficientdet_d2_coco17_tpu-32/checkpoint/ckpt-0'


with open(pipeline_config_path, 'r') as f:
    config = f.read()

# Modify the configuration
config = re.sub('label_map_path: ".*?"',
                'label_map_path: "{}"'.format(labelmap_path), config)

config = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)

config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
                'input_path: "{}"'.format(train_record_path), config)

config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
                'input_path: "{}"'.format(test_record_path), config)

config = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), config)

config = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), config)

config = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), config)

config = re.sub(r'learning_rate_base:\s*\d*\.?\d+', f'learning_rate_base: {learning_rate_base}', config)
config = re.sub(r'total_steps:\s*\d+', f'total_steps: {total_steps}', config)
config = re.sub(r'warmup_learning_rate:\s*\d*\.?\d+', f'warmup_learning_rate: {warmup_learning_rate}', config)
config = re.sub(r'warmup_steps:\s*\d+', f'warmup_steps: {warmup_steps}', config)

config = re.sub('fine_tune_checkpoint_type: "classification"',
                'fine_tune_checkpoint_type: "{}"'.format(checkpoint_type), config)



print("\nModified Config:\n", config)
# Write the updated configuration back to the file
with open(pipeline_config_path, 'w') as f:
    f.write(config)
