�	��s��@��s��@!��s��@	=��_9I@=��_9I@!=��_9I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��s��@���Jp�?A�%P6E@Y�`<���?*	E����|b@2F
Iterator::Modelc{-�1�?!W��8�O@)�T�-��?1v@��=�G@:Preprocessing2U
Iterator::Model::ParallelMapV2�� �> �?!M-C��0@)�� �> �?1M-C��0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatKr��&O�?!%��0@).=�����?13���	e*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�1��l�?!��O��.@)���T��?1a*!��s$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceOt	�~?!Lu�9(@)Ot	�~?1Lu�9(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�n�;2V�?!�w�B@)���C�x?1��[�~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorxԘsIu?!s"�V@)xԘsIu?1s"�V@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 37.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9=��_9I@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���Jp�?���Jp�?!���Jp�?      ��!       "      ��!       *      ��!       2	�%P6E@�%P6E@!�%P6E@:      ��!       B      ��!       J	�`<���?�`<���?!�`<���?R      ��!       Z	�`<���?�`<���?!�`<���?JCPU_ONLYY=��_9I@b Y      Y@qR�&|�?Q@"�
both�Your program is POTENTIALLY input-bound because 37.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb�68.9934% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 