
4
PlaceholderPlaceholder*
dtype0*
shape: 
�
6hidden_layer/1/weights/Initializer/random_normal/shapeConst*
dtype0*%
valueB"         �   *)
_class
loc:@hidden_layer/1/weights
�
5hidden_layer/1/weights/Initializer/random_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@hidden_layer/1/weights
�
7hidden_layer/1/weights/Initializer/random_normal/stddevConst*
dtype0*
valueB
 *��=*)
_class
loc:@hidden_layer/1/weights
�
Ehidden_layer/1/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6hidden_layer/1/weights/Initializer/random_normal/shape*
dtype0*

seed *)
_class
loc:@hidden_layer/1/weights*
seed2 *
T0
�
4hidden_layer/1/weights/Initializer/random_normal/mulMulEhidden_layer/1/weights/Initializer/random_normal/RandomStandardNormal7hidden_layer/1/weights/Initializer/random_normal/stddev*)
_class
loc:@hidden_layer/1/weights*
T0
�
0hidden_layer/1/weights/Initializer/random_normalAdd4hidden_layer/1/weights/Initializer/random_normal/mul5hidden_layer/1/weights/Initializer/random_normal/mean*)
_class
loc:@hidden_layer/1/weights*
T0
�
hidden_layer/1/weights
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@hidden_layer/1/weights*
shape:�
�
hidden_layer/1/weights/AssignAssignhidden_layer/1/weights0hidden_layer/1/weights/Initializer/random_normal*
T0*
validate_shape(*)
_class
loc:@hidden_layer/1/weights*
use_locking(
s
hidden_layer/1/weights/readIdentityhidden_layer/1/weights*)
_class
loc:@hidden_layer/1/weights*
T0

%hidden_layer/1/bias/Initializer/ConstConst*
dtype0*
valueB�*
�#<*&
_class
loc:@hidden_layer/1/bias
�
hidden_layer/1/bias
VariableV2*
dtype0*
shared_name *
	container *&
_class
loc:@hidden_layer/1/bias*
shape:�
�
hidden_layer/1/bias/AssignAssignhidden_layer/1/bias%hidden_layer/1/bias/Initializer/Const*
T0*
validate_shape(*&
_class
loc:@hidden_layer/1/bias*
use_locking(
j
hidden_layer/1/bias/readIdentityhidden_layer/1/bias*&
_class
loc:@hidden_layer/1/bias*
T0
�
hidden_layer/1/Conv2DConv2DPlaceholderhidden_layer/1/weights/read*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
r
hidden_layer/1/BiasAddBiasAddhidden_layer/1/Conv2Dhidden_layer/1/bias/read*
T0*
data_formatNHWC
<
hidden_layer/1/ReluReluhidden_layer/1/BiasAdd*
T0
�
6hidden_layer/2/weights/Initializer/random_normal/shapeConst*
dtype0*%
valueB"      �   �   *)
_class
loc:@hidden_layer/2/weights
�
5hidden_layer/2/weights/Initializer/random_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@hidden_layer/2/weights
�
7hidden_layer/2/weights/Initializer/random_normal/stddevConst*
dtype0*
valueB
 *�Q<*)
_class
loc:@hidden_layer/2/weights
�
Ehidden_layer/2/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6hidden_layer/2/weights/Initializer/random_normal/shape*
dtype0*

seed *)
_class
loc:@hidden_layer/2/weights*
seed2 *
T0
�
4hidden_layer/2/weights/Initializer/random_normal/mulMulEhidden_layer/2/weights/Initializer/random_normal/RandomStandardNormal7hidden_layer/2/weights/Initializer/random_normal/stddev*)
_class
loc:@hidden_layer/2/weights*
T0
�
0hidden_layer/2/weights/Initializer/random_normalAdd4hidden_layer/2/weights/Initializer/random_normal/mul5hidden_layer/2/weights/Initializer/random_normal/mean*)
_class
loc:@hidden_layer/2/weights*
T0
�
hidden_layer/2/weights
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@hidden_layer/2/weights*
shape:��
�
hidden_layer/2/weights/AssignAssignhidden_layer/2/weights0hidden_layer/2/weights/Initializer/random_normal*
T0*
validate_shape(*)
_class
loc:@hidden_layer/2/weights*
use_locking(
s
hidden_layer/2/weights/readIdentityhidden_layer/2/weights*)
_class
loc:@hidden_layer/2/weights*
T0

%hidden_layer/2/bias/Initializer/ConstConst*
dtype0*
valueB�*
�#<*&
_class
loc:@hidden_layer/2/bias
�
hidden_layer/2/bias
VariableV2*
dtype0*
shared_name *
	container *&
_class
loc:@hidden_layer/2/bias*
shape:�
�
hidden_layer/2/bias/AssignAssignhidden_layer/2/bias%hidden_layer/2/bias/Initializer/Const*
T0*
validate_shape(*&
_class
loc:@hidden_layer/2/bias*
use_locking(
j
hidden_layer/2/bias/readIdentityhidden_layer/2/bias*&
_class
loc:@hidden_layer/2/bias*
T0
�
hidden_layer/2/Conv2DConv2Dhidden_layer/1/Reluhidden_layer/2/weights/read*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
r
hidden_layer/2/BiasAddBiasAddhidden_layer/2/Conv2Dhidden_layer/2/bias/read*
T0*
data_formatNHWC
<
hidden_layer/2/ReluReluhidden_layer/2/BiasAdd*
T0
�
6hidden_layer/3/weights/Initializer/random_normal/shapeConst*
dtype0*%
valueB"      �   �   *)
_class
loc:@hidden_layer/3/weights
�
5hidden_layer/3/weights/Initializer/random_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@hidden_layer/3/weights
�
7hidden_layer/3/weights/Initializer/random_normal/stddevConst*
dtype0*
valueB
 *�Q<*)
_class
loc:@hidden_layer/3/weights
�
Ehidden_layer/3/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6hidden_layer/3/weights/Initializer/random_normal/shape*
dtype0*

seed *)
_class
loc:@hidden_layer/3/weights*
seed2 *
T0
�
4hidden_layer/3/weights/Initializer/random_normal/mulMulEhidden_layer/3/weights/Initializer/random_normal/RandomStandardNormal7hidden_layer/3/weights/Initializer/random_normal/stddev*)
_class
loc:@hidden_layer/3/weights*
T0
�
0hidden_layer/3/weights/Initializer/random_normalAdd4hidden_layer/3/weights/Initializer/random_normal/mul5hidden_layer/3/weights/Initializer/random_normal/mean*)
_class
loc:@hidden_layer/3/weights*
T0
�
hidden_layer/3/weights
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@hidden_layer/3/weights*
shape:��
�
hidden_layer/3/weights/AssignAssignhidden_layer/3/weights0hidden_layer/3/weights/Initializer/random_normal*
T0*
validate_shape(*)
_class
loc:@hidden_layer/3/weights*
use_locking(
s
hidden_layer/3/weights/readIdentityhidden_layer/3/weights*)
_class
loc:@hidden_layer/3/weights*
T0

%hidden_layer/3/bias/Initializer/ConstConst*
dtype0*
valueB�*
�#<*&
_class
loc:@hidden_layer/3/bias
�
hidden_layer/3/bias
VariableV2*
dtype0*
shared_name *
	container *&
_class
loc:@hidden_layer/3/bias*
shape:�
�
hidden_layer/3/bias/AssignAssignhidden_layer/3/bias%hidden_layer/3/bias/Initializer/Const*
T0*
validate_shape(*&
_class
loc:@hidden_layer/3/bias*
use_locking(
j
hidden_layer/3/bias/readIdentityhidden_layer/3/bias*&
_class
loc:@hidden_layer/3/bias*
T0
�
hidden_layer/3/Conv2DConv2Dhidden_layer/2/Reluhidden_layer/3/weights/read*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
r
hidden_layer/3/BiasAddBiasAddhidden_layer/3/Conv2Dhidden_layer/3/bias/read*
T0*
data_formatNHWC
<
hidden_layer/3/ReluReluhidden_layer/3/BiasAdd*
T0
�
6hidden_layer/4/weights/Initializer/random_normal/shapeConst*
dtype0*%
valueB"      �   �   *)
_class
loc:@hidden_layer/4/weights
�
5hidden_layer/4/weights/Initializer/random_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@hidden_layer/4/weights
�
7hidden_layer/4/weights/Initializer/random_normal/stddevConst*
dtype0*
valueB
 *�Q<*)
_class
loc:@hidden_layer/4/weights
�
Ehidden_layer/4/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6hidden_layer/4/weights/Initializer/random_normal/shape*
dtype0*

seed *)
_class
loc:@hidden_layer/4/weights*
seed2 *
T0
�
4hidden_layer/4/weights/Initializer/random_normal/mulMulEhidden_layer/4/weights/Initializer/random_normal/RandomStandardNormal7hidden_layer/4/weights/Initializer/random_normal/stddev*)
_class
loc:@hidden_layer/4/weights*
T0
�
0hidden_layer/4/weights/Initializer/random_normalAdd4hidden_layer/4/weights/Initializer/random_normal/mul5hidden_layer/4/weights/Initializer/random_normal/mean*)
_class
loc:@hidden_layer/4/weights*
T0
�
hidden_layer/4/weights
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@hidden_layer/4/weights*
shape:��
�
hidden_layer/4/weights/AssignAssignhidden_layer/4/weights0hidden_layer/4/weights/Initializer/random_normal*
T0*
validate_shape(*)
_class
loc:@hidden_layer/4/weights*
use_locking(
s
hidden_layer/4/weights/readIdentityhidden_layer/4/weights*)
_class
loc:@hidden_layer/4/weights*
T0

%hidden_layer/4/bias/Initializer/ConstConst*
dtype0*
valueB�*
�#<*&
_class
loc:@hidden_layer/4/bias
�
hidden_layer/4/bias
VariableV2*
dtype0*
shared_name *
	container *&
_class
loc:@hidden_layer/4/bias*
shape:�
�
hidden_layer/4/bias/AssignAssignhidden_layer/4/bias%hidden_layer/4/bias/Initializer/Const*
T0*
validate_shape(*&
_class
loc:@hidden_layer/4/bias*
use_locking(
j
hidden_layer/4/bias/readIdentityhidden_layer/4/bias*&
_class
loc:@hidden_layer/4/bias*
T0
�
hidden_layer/4/Conv2DConv2Dhidden_layer/3/Reluhidden_layer/4/weights/read*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
r
hidden_layer/4/BiasAddBiasAddhidden_layer/4/Conv2Dhidden_layer/4/bias/read*
T0*
data_formatNHWC
<
hidden_layer/4/ReluReluhidden_layer/4/BiasAdd*
T0
�
6hidden_layer/5/weights/Initializer/random_normal/shapeConst*
dtype0*%
valueB"      �      *)
_class
loc:@hidden_layer/5/weights
�
5hidden_layer/5/weights/Initializer/random_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@hidden_layer/5/weights
�
7hidden_layer/5/weights/Initializer/random_normal/stddevConst*
dtype0*
valueB
 *�Q<*)
_class
loc:@hidden_layer/5/weights
�
Ehidden_layer/5/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6hidden_layer/5/weights/Initializer/random_normal/shape*
dtype0*

seed *)
_class
loc:@hidden_layer/5/weights*
seed2 *
T0
�
4hidden_layer/5/weights/Initializer/random_normal/mulMulEhidden_layer/5/weights/Initializer/random_normal/RandomStandardNormal7hidden_layer/5/weights/Initializer/random_normal/stddev*)
_class
loc:@hidden_layer/5/weights*
T0
�
0hidden_layer/5/weights/Initializer/random_normalAdd4hidden_layer/5/weights/Initializer/random_normal/mul5hidden_layer/5/weights/Initializer/random_normal/mean*)
_class
loc:@hidden_layer/5/weights*
T0
�
hidden_layer/5/weights
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@hidden_layer/5/weights*
shape:�
�
hidden_layer/5/weights/AssignAssignhidden_layer/5/weights0hidden_layer/5/weights/Initializer/random_normal*
T0*
validate_shape(*)
_class
loc:@hidden_layer/5/weights*
use_locking(
s
hidden_layer/5/weights/readIdentityhidden_layer/5/weights*)
_class
loc:@hidden_layer/5/weights*
T0
~
%hidden_layer/5/bias/Initializer/ConstConst*
dtype0*
valueB*
�#<*&
_class
loc:@hidden_layer/5/bias
�
hidden_layer/5/bias
VariableV2*
dtype0*
shared_name *
	container *&
_class
loc:@hidden_layer/5/bias*
shape:
�
hidden_layer/5/bias/AssignAssignhidden_layer/5/bias%hidden_layer/5/bias/Initializer/Const*
T0*
validate_shape(*&
_class
loc:@hidden_layer/5/bias*
use_locking(
j
hidden_layer/5/bias/readIdentityhidden_layer/5/bias*&
_class
loc:@hidden_layer/5/bias*
T0
�
hidden_layer/5/Conv2DConv2Dhidden_layer/4/Reluhidden_layer/5/weights/read*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
r
hidden_layer/5/BiasAddBiasAddhidden_layer/5/Conv2Dhidden_layer/5/bias/read*
T0*
data_formatNHWC
<
hidden_layer/5/ReluReluhidden_layer/5/BiasAdd*
T0
B
Reshape/shapeConst*
dtype0*
valueB"����@   
M
ReshapeReshapehidden_layer/5/ReluReshape/shape*
Tshape0*
T0
�
5output_layer/weights/Initializer/random_uniform/shapeConst*
dtype0*
valueB"@   �   *'
_class
loc:@output_layer/weights
�
3output_layer/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *b~)�*'
_class
loc:@output_layer/weights
�
3output_layer/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *b~)>*'
_class
loc:@output_layer/weights
�
=output_layer/weights/Initializer/random_uniform/RandomUniformRandomUniform5output_layer/weights/Initializer/random_uniform/shape*
dtype0*

seed *'
_class
loc:@output_layer/weights*
seed2 *
T0
�
3output_layer/weights/Initializer/random_uniform/subSub3output_layer/weights/Initializer/random_uniform/max3output_layer/weights/Initializer/random_uniform/min*'
_class
loc:@output_layer/weights*
T0
�
3output_layer/weights/Initializer/random_uniform/mulMul=output_layer/weights/Initializer/random_uniform/RandomUniform3output_layer/weights/Initializer/random_uniform/sub*'
_class
loc:@output_layer/weights*
T0
�
/output_layer/weights/Initializer/random_uniformAdd3output_layer/weights/Initializer/random_uniform/mul3output_layer/weights/Initializer/random_uniform/min*'
_class
loc:@output_layer/weights*
T0
�
output_layer/weights
VariableV2*
dtype0*
shared_name *
	container *'
_class
loc:@output_layer/weights*
shape:	@�
�
output_layer/weights/AssignAssignoutput_layer/weights/output_layer/weights/Initializer/random_uniform*
T0*
validate_shape(*'
_class
loc:@output_layer/weights*
use_locking(
m
output_layer/weights/readIdentityoutput_layer/weights*'
_class
loc:@output_layer/weights*
T0
{
#output_layer/bias/Initializer/ConstConst*
dtype0*
valueB�*    *$
_class
loc:@output_layer/bias
�
output_layer/bias
VariableV2*
dtype0*
shared_name *
	container *$
_class
loc:@output_layer/bias*
shape:�
�
output_layer/bias/AssignAssignoutput_layer/bias#output_layer/bias/Initializer/Const*
T0*
validate_shape(*$
_class
loc:@output_layer/bias*
use_locking(
d
output_layer/bias/readIdentityoutput_layer/bias*$
_class
loc:@output_layer/bias*
T0
p
output_layer/MatMulMatMulReshapeoutput_layer/weights/read*
transpose_b( *
transpose_a( *
T0
k
output_layer/outputBiasAddoutput_layer/MatMuloutput_layer/bias/read*
T0*
data_formatNHWC
=
output_layer/SoftmaxSoftmaxoutput_layer/output*
T0
8

save/ConstConst*
dtype0*
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�Bhidden_layer/1/biasBhidden_layer/1/weightsBhidden_layer/2/biasBhidden_layer/2/weightsBhidden_layer/3/biasBhidden_layer/3/weightsBhidden_layer/4/biasBhidden_layer/4/weightsBhidden_layer/5/biasBhidden_layer/5/weightsBoutput_layer/biasBoutput_layer/weights
_
save/SaveV2/shape_and_slicesConst*
dtype0*+
value"B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceshidden_layer/1/biashidden_layer/1/weightshidden_layer/2/biashidden_layer/2/weightshidden_layer/3/biashidden_layer/3/weightshidden_layer/4/biashidden_layer/4/weightshidden_layer/5/biashidden_layer/5/weightsoutput_layer/biasoutput_layer/weights*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0
[
save/RestoreV2/tensor_namesConst*
dtype0*(
valueBBhidden_layer/1/bias
L
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
�
save/AssignAssignhidden_layer/1/biassave/RestoreV2*
T0*
validate_shape(*&
_class
loc:@hidden_layer/1/bias*
use_locking(
`
save/RestoreV2_1/tensor_namesConst*
dtype0*+
value"B Bhidden_layer/1/weights
N
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2
�
save/Assign_1Assignhidden_layer/1/weightssave/RestoreV2_1*
T0*
validate_shape(*)
_class
loc:@hidden_layer/1/weights*
use_locking(
]
save/RestoreV2_2/tensor_namesConst*
dtype0*(
valueBBhidden_layer/2/bias
N
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2
�
save/Assign_2Assignhidden_layer/2/biassave/RestoreV2_2*
T0*
validate_shape(*&
_class
loc:@hidden_layer/2/bias*
use_locking(
`
save/RestoreV2_3/tensor_namesConst*
dtype0*+
value"B Bhidden_layer/2/weights
N
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2
�
save/Assign_3Assignhidden_layer/2/weightssave/RestoreV2_3*
T0*
validate_shape(*)
_class
loc:@hidden_layer/2/weights*
use_locking(
]
save/RestoreV2_4/tensor_namesConst*
dtype0*(
valueBBhidden_layer/3/bias
N
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
�
save/Assign_4Assignhidden_layer/3/biassave/RestoreV2_4*
T0*
validate_shape(*&
_class
loc:@hidden_layer/3/bias*
use_locking(
`
save/RestoreV2_5/tensor_namesConst*
dtype0*+
value"B Bhidden_layer/3/weights
N
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2
�
save/Assign_5Assignhidden_layer/3/weightssave/RestoreV2_5*
T0*
validate_shape(*)
_class
loc:@hidden_layer/3/weights*
use_locking(
]
save/RestoreV2_6/tensor_namesConst*
dtype0*(
valueBBhidden_layer/4/bias
N
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2
�
save/Assign_6Assignhidden_layer/4/biassave/RestoreV2_6*
T0*
validate_shape(*&
_class
loc:@hidden_layer/4/bias*
use_locking(
`
save/RestoreV2_7/tensor_namesConst*
dtype0*+
value"B Bhidden_layer/4/weights
N
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2
�
save/Assign_7Assignhidden_layer/4/weightssave/RestoreV2_7*
T0*
validate_shape(*)
_class
loc:@hidden_layer/4/weights*
use_locking(
]
save/RestoreV2_8/tensor_namesConst*
dtype0*(
valueBBhidden_layer/5/bias
N
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
�
save/Assign_8Assignhidden_layer/5/biassave/RestoreV2_8*
T0*
validate_shape(*&
_class
loc:@hidden_layer/5/bias*
use_locking(
`
save/RestoreV2_9/tensor_namesConst*
dtype0*+
value"B Bhidden_layer/5/weights
N
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
�
save/Assign_9Assignhidden_layer/5/weightssave/RestoreV2_9*
T0*
validate_shape(*)
_class
loc:@hidden_layer/5/weights*
use_locking(
\
save/RestoreV2_10/tensor_namesConst*
dtype0*&
valueBBoutput_layer/bias
O
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2
�
save/Assign_10Assignoutput_layer/biassave/RestoreV2_10*
T0*
validate_shape(*$
_class
loc:@output_layer/bias*
use_locking(
_
save/RestoreV2_11/tensor_namesConst*
dtype0*)
value BBoutput_layer/weights
O
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2
�
save/Assign_11Assignoutput_layer/weightssave/RestoreV2_11*
T0*
validate_shape(*'
_class
loc:@output_layer/weights*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11"