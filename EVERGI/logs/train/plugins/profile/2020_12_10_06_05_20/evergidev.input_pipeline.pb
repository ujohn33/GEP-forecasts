$	�\����?"s�+���?d@�z�ǣ?!�G7¢�?	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�E(����?:u�<�?A}A	]�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsd@�z�ǣ?���ꫫ�?A6���a?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�G7¢�?"��`�?Aa2U0*��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsePmp"��?������?Ah^��^?*	���x�d@2F
Iterator::Modelı.n��?!d[���P@)v���z�?1����ՋI@:Preprocessing2U
Iterator::Model::ParallelMapV2:!t�%�?!,t�q�/@):!t�%�?1,t�q�/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�a����?!�+���W-@)�э����?1���ș;'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��=z�}�?!Ēx�_(@)�a��A�?1�ѥ�`� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�V����?!8I�ʱ�@@).�R\U�}?1��e,h�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��[�ty?!�KO�F@)��[�ty?1�KO�F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorh�N?��t?!Y�,Ϥo@)h�N?��t?1Y�,Ϥo@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�n�e���?!�%��,@)���[�k?1]ıNƍ @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 46.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	:�S���?�m�!��?���ꫫ�?!������?	!       "	!       *	!       2$	%�\R�ݱ?IO{�G�?h^��^?!a2U0*��?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 