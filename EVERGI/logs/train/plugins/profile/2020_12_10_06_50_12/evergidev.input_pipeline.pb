$	ͰQ�o��?������?����g�?!��rf{�?	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails���/J��?���"[�?A�A�<�E�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��rf{�?}[�Tp�?A�C��<�f?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails����?�)t^c��?A�����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails����g�?.�s`9�?A�7k�*g?*	 �rh�t@2F
Iterator::ModelLnYk(�?!��DC,P@)��L��?16�|G�G@:Preprocessing2U
Iterator::Model::ParallelMapV2�X32�]�?!}V�~�0@)�X32�]�?1}V�~�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�W歺�?!g����M1@)��;�%�?1S�k���,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatek:!tХ?!�S}=�)@)�g�����?1u�!��� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�a�Q+L�?!��b�[�@)�a�Q+L�?1��b�[�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipc��*3��?!���vy�A@)(�r�w�?1r���T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��5�e��?!�e�\�c@)��5�e��?1�e�\�c@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
MK�ݧ?!�C�l,@)��N�jp?1�G4���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	Y��0_�?�1���?���"[�?!}[�Tp�?	!       "	!       *	!       2$	I�V
��?�7,X�?�C��<�f?!�A�<�E�?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 