�	�@�"y@�@�"y@!�@�"y@	nQ����@nQ����@!nQ����@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�@�"y@Քd���?A���·g@Yc��*3��?*	�~j�t�b@2F
Iterator::Model��H/j��?!U�U�ߚO@)�	�5��?1CMTqF@:Preprocessing2U
Iterator::Model::ParallelMapV2X�B�_˛?!#��S2@)X�B�_˛?1#��S2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Ma���?!!
��z1@)�c[���?1*�M�8:+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapcC7��?!�*�1.@)�����Ս?1�Oq��#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice* �3h�?!ɵ9W�	@)* �3h�?1ɵ9W�	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���q��?!�4�2 eB@)�>�6�y?1�'ӣ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorvS�k%tw?!\�1���@)vS�k%tw?1\�1���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9nQ����@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Քd���?Քd���?!Քd���?      ��!       "      ��!       *      ��!       2	���·g@���·g@!���·g@:      ��!       B      ��!       J	c��*3��?c��*3��?!c��*3��?R      ��!       Z	c��*3��?c��*3��?!c��*3��?JCPU_ONLYYnQ����@b Y      Y@q�49f�T@"�
both�Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb�83.2094% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 