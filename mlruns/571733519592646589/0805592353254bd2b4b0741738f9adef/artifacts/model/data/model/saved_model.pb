��:
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��1
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:�*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_3/bias
y
(Adam/v/conv1d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_3/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_3/bias
y
(Adam/m/conv1d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_3/kernel
�
*Adam/v/conv1d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_3/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_3/kernel
�
*Adam/m/conv1d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_3/kernel*"
_output_shapes
:*
dtype0
�
!Adam/v/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_7/beta
�
5Adam/v/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_7/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_7/beta
�
5Adam/m/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_7/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_7/gamma
�
6Adam/v/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_7/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_7/gamma
�
6Adam/m/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_7/gamma*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/v/multi_head_attention_3/attention_output/bias
�
GAdam/v/multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention_3/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/m/multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/m/multi_head_attention_3/attention_output/bias
�
GAdam/m/multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention_3/attention_output/bias*
_output_shapes
:*
dtype0
�
5Adam/v/multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/v/multi_head_attention_3/attention_output/kernel
�
IAdam/v/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/v/multi_head_attention_3/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
5Adam/m/multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/m/multi_head_attention_3/attention_output/kernel
�
IAdam/m/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/m/multi_head_attention_3/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_3/value/bias
�
<Adam/v/multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_3/value/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_3/value/bias
�
<Adam/m/multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_3/value/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_3/value/kernel
�
>Adam/v/multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_3/value/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_3/value/kernel
�
>Adam/m/multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_3/value/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/v/multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/multi_head_attention_3/key/bias
�
:Adam/v/multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention_3/key/bias*
_output_shapes

:@*
dtype0
�
&Adam/m/multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/multi_head_attention_3/key/bias
�
:Adam/m/multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention_3/key/bias*
_output_shapes

:@*
dtype0
�
(Adam/v/multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/multi_head_attention_3/key/kernel
�
<Adam/v/multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_3/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/m/multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/multi_head_attention_3/key/kernel
�
<Adam/m/multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_3/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_3/query/bias
�
<Adam/v/multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_3/query/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_3/query/bias
�
<Adam/m/multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_3/query/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_3/query/kernel
�
>Adam/v/multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_3/query/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_3/query/kernel
�
>Adam/m/multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_3/query/kernel*"
_output_shapes
:@*
dtype0
�
!Adam/v/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_6/beta
�
5Adam/v/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_6/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_6/beta
�
5Adam/m/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_6/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_6/gamma
�
6Adam/v/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_6/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_6/gamma
�
6Adam/m/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_6/gamma*
_output_shapes
:*
dtype0
�
Adam/v/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_2/bias
y
(Adam/v/conv1d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_2/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_2/bias
y
(Adam/m/conv1d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_2/kernel
�
*Adam/v/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_2/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_2/kernel
�
*Adam/m/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_2/kernel*"
_output_shapes
:*
dtype0
�
!Adam/v/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_5/beta
�
5Adam/v/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_5/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_5/beta
�
5Adam/m/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_5/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_5/gamma
�
6Adam/v/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_5/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_5/gamma
�
6Adam/m/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_5/gamma*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/v/multi_head_attention_2/attention_output/bias
�
GAdam/v/multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention_2/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/m/multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/m/multi_head_attention_2/attention_output/bias
�
GAdam/m/multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention_2/attention_output/bias*
_output_shapes
:*
dtype0
�
5Adam/v/multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/v/multi_head_attention_2/attention_output/kernel
�
IAdam/v/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/v/multi_head_attention_2/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
5Adam/m/multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/m/multi_head_attention_2/attention_output/kernel
�
IAdam/m/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/m/multi_head_attention_2/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_2/value/bias
�
<Adam/v/multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_2/value/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_2/value/bias
�
<Adam/m/multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_2/value/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_2/value/kernel
�
>Adam/v/multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_2/value/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_2/value/kernel
�
>Adam/m/multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_2/value/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/v/multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/multi_head_attention_2/key/bias
�
:Adam/v/multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention_2/key/bias*
_output_shapes

:@*
dtype0
�
&Adam/m/multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/multi_head_attention_2/key/bias
�
:Adam/m/multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention_2/key/bias*
_output_shapes

:@*
dtype0
�
(Adam/v/multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/multi_head_attention_2/key/kernel
�
<Adam/v/multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_2/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/m/multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/multi_head_attention_2/key/kernel
�
<Adam/m/multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_2/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_2/query/bias
�
<Adam/v/multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_2/query/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_2/query/bias
�
<Adam/m/multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_2/query/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_2/query/kernel
�
>Adam/v/multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_2/query/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_2/query/kernel
�
>Adam/m/multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_2/query/kernel*"
_output_shapes
:@*
dtype0
�
!Adam/v/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_4/beta
�
5Adam/v/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_4/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_4/beta
�
5Adam/m/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_4/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_4/gamma
�
6Adam/v/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_4/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_4/gamma
�
6Adam/m/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_4/gamma*
_output_shapes
:*
dtype0
�
Adam/v/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_1/bias
y
(Adam/v/conv1d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_1/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_1/bias
y
(Adam/m/conv1d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_1/kernel
�
*Adam/v/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_1/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_1/kernel
�
*Adam/m/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_1/kernel*"
_output_shapes
:*
dtype0
�
!Adam/v/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_3/beta
�
5Adam/v/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_3/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_3/beta
�
5Adam/m/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_3/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_3/gamma
�
6Adam/v/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_3/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_3/gamma
�
6Adam/m/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_3/gamma*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/v/multi_head_attention_1/attention_output/bias
�
GAdam/v/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention_1/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/m/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/m/multi_head_attention_1/attention_output/bias
�
GAdam/m/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention_1/attention_output/bias*
_output_shapes
:*
dtype0
�
5Adam/v/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/v/multi_head_attention_1/attention_output/kernel
�
IAdam/v/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/v/multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
5Adam/m/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/m/multi_head_attention_1/attention_output/kernel
�
IAdam/m/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/m/multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_1/value/bias
�
<Adam/v/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_1/value/bias
�
<Adam/m/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_1/value/kernel
�
>Adam/v/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_1/value/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_1/value/kernel
�
>Adam/m/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_1/value/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/v/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/multi_head_attention_1/key/bias
�
:Adam/v/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
�
&Adam/m/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/multi_head_attention_1/key/bias
�
:Adam/m/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
�
(Adam/v/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/multi_head_attention_1/key/kernel
�
<Adam/v/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_1/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/m/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/multi_head_attention_1/key/kernel
�
<Adam/m/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_1/key/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/v/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/v/multi_head_attention_1/query/bias
�
<Adam/v/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
�
(Adam/m/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(Adam/m/multi_head_attention_1/query/bias
�
<Adam/m/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
�
*Adam/v/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/v/multi_head_attention_1/query/kernel
�
>Adam/v/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_1/query/kernel*"
_output_shapes
:@*
dtype0
�
*Adam/m/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/m/multi_head_attention_1/query/kernel
�
>Adam/m/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_1/query/kernel*"
_output_shapes
:@*
dtype0
�
!Adam/v/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_2/beta
�
5Adam/v/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_2/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_2/beta
�
5Adam/m/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_2/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_2/gamma
�
6Adam/v/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_2/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_2/gamma
�
6Adam/m/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_2/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv1d/bias
u
&Adam/v/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv1d/bias
u
&Adam/m/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d/kernel
�
(Adam/v/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d/kernel
�
(Adam/m/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/kernel*"
_output_shapes
:*
dtype0
�
!Adam/v/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_1/beta
�
5Adam/v/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_1/beta*
_output_shapes
:*
dtype0
�
!Adam/m/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_1/beta
�
5Adam/m/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_1/beta*
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_1/gamma
�
6Adam/v/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_1/gamma
�
6Adam/m/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
�
1Adam/v/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/v/multi_head_attention/attention_output/bias
�
EAdam/v/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp1Adam/v/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
�
1Adam/m/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/m/multi_head_attention/attention_output/bias
�
EAdam/m/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp1Adam/m/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/v/multi_head_attention/attention_output/kernel
�
GAdam/v/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
3Adam/m/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/m/multi_head_attention/attention_output/kernel
�
GAdam/m/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/v/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/multi_head_attention/value/bias
�
:Adam/v/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
&Adam/m/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/multi_head_attention/value/bias
�
:Adam/m/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
(Adam/v/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/multi_head_attention/value/kernel
�
<Adam/v/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention/value/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/m/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/multi_head_attention/value/kernel
�
<Adam/m/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention/value/kernel*"
_output_shapes
:@*
dtype0
�
$Adam/v/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$Adam/v/multi_head_attention/key/bias
�
8Adam/v/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp$Adam/v/multi_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
$Adam/m/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$Adam/m/multi_head_attention/key/bias
�
8Adam/m/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp$Adam/m/multi_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
&Adam/v/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/v/multi_head_attention/key/kernel
�
:Adam/v/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/key/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/m/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/m/multi_head_attention/key/kernel
�
:Adam/m/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/key/kernel*"
_output_shapes
:@*
dtype0
�
&Adam/v/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/multi_head_attention/query/bias
�
:Adam/v/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
&Adam/m/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/multi_head_attention/query/bias
�
:Adam/m/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
(Adam/v/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/multi_head_attention/query/kernel
�
<Adam/v/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention/query/kernel*"
_output_shapes
:@*
dtype0
�
(Adam/m/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/multi_head_attention/query/kernel
�
<Adam/m/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention/query/kernel*"
_output_shapes
:@*
dtype0
�
Adam/v/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/layer_normalization/beta
�
3Adam/v/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/layer_normalization/beta*
_output_shapes
:*
dtype0
�
Adam/m/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/layer_normalization/beta
�
3Adam/m/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/layer_normalization/beta*
_output_shapes
:*
dtype0
�
 Adam/v/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/layer_normalization/gamma
�
4Adam/v/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/layer_normalization/gamma*
_output_shapes
:*
dtype0
�
 Adam/m/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/layer_normalization/gamma
�
4Adam/m/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/layer_normalization/gamma*
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
,multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_3/attention_output/bias
�
@multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_3/attention_output/bias*
_output_shapes
:*
dtype0
�
.multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.multi_head_attention_3/attention_output/kernel
�
Bmulti_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_3/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_3/value/bias
�
5multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/value/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_3/value/kernel
�
7multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_3/value/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_3/key/bias
�
3multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_3/key/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!multi_head_attention_3/key/kernel
�
5multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/key/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_3/query/bias
�
5multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/query/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_3/query/kernel
�
7multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_3/query/kernel*"
_output_shapes
:@*
dtype0
�
,multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_2/attention_output/bias
�
@multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_2/attention_output/bias*
_output_shapes
:*
dtype0
�
.multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.multi_head_attention_2/attention_output/kernel
�
Bmulti_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_2/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_2/value/bias
�
5multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/value/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_2/value/kernel
�
7multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/value/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_2/key/bias
�
3multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_2/key/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!multi_head_attention_2/key/kernel
�
5multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/key/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_2/query/bias
�
5multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/query/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_2/query/kernel
�
7multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/query/kernel*"
_output_shapes
:@*
dtype0
�
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_1/attention_output/bias
�
@multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_1/attention_output/bias*
_output_shapes
:*
dtype0
�
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.multi_head_attention_1/attention_output/kernel
�
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/value/bias
�
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_1/value/kernel
�
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_1/key/bias
�
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!multi_head_attention_1/key/kernel
�
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:@*
dtype0
�
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/query/bias
�
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#multi_head_attention_1/query/kernel
�
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:@*
dtype0
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@*
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:*
dtype0
�
layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_7/beta
�
.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
:*
dtype0
�
layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_7/gamma
�
/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
:*
dtype0
�
layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_6/beta
�
.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:*
dtype0
�
layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_6/gamma
�
/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
�
layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_5/beta
�
.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:*
dtype0
�
layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_5/gamma
�
/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:*
dtype0
�
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_4/beta
�
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:*
dtype0
�
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_4/gamma
�
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
�
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_3/beta
�
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:*
dtype0
�
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_3/gamma
�
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:*
dtype0
�
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_2/beta
�
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:*
dtype0
�
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_2/gamma
�
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
�
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_1/beta
�
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:*
dtype0
�
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_1/gamma
�
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:*
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biaslayer_normalization_2/gammalayer_normalization_2/beta#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biaslayer_normalization_3/gammalayer_normalization_3/betaconv1d_1/kernelconv1d_1/biaslayer_normalization_4/gammalayer_normalization_4/beta#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/biaslayer_normalization_5/gammalayer_normalization_5/betaconv1d_2/kernelconv1d_2/biaslayer_normalization_6/gammalayer_normalization_6/beta#multi_head_attention_3/query/kernel!multi_head_attention_3/query/bias!multi_head_attention_3/key/kernelmulti_head_attention_3/key/bias#multi_head_attention_3/value/kernel!multi_head_attention_3/value/bias.multi_head_attention_3/attention_output/kernel,multi_head_attention_3/attention_output/biaslayer_normalization_7/gammalayer_normalization_7/betaconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_238718

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_default_save_signature
)	optimizer
*
signatures*
* 
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1axis
	2gamma
3beta*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_query_dense
;
_key_dense
<_value_dense
=_softmax
>_dropout_layer
?_output_dense*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator* 
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_query_dense
u
_key_dense
v_value_dense
w_softmax
x_dropout_layer
y_output_dense*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
20
31
�2
�3
�4
�5
�6
�7
�8
�9
M10
N11
U12
V13
l14
m15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59*
�
20
31
�2
�3
�4
�5
�6
�7
�8
�9
M10
N11
U12
V13
l14
m15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
(_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

l0
m1*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv1d_2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_6/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_6/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_7/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_7/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv1d_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmulti_head_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,multi_head_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*multi_head_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_1/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/query/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/query/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/key/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_2/key/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/value/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/value/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_2/attention_output/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_2/attention_output/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_3/query/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/query/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/key/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_3/key/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_3/value/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/value/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_3/attention_output/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_3/attention_output/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92
�93
�94
�95
�96
�97
�98
�99
�100
�101
�102
�103
�104
�105
�106
�107
�108
�109
�110
�111
�112
�113
�114
�115
�116
�117
�118
�119
�120*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
:0
;1
<2
=3
>4
?5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
t0
u1
v2
w3
x4
y5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
ke
VARIABLE_VALUE Adam/m/layer_normalization/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/layer_normalization/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/layer_normalization/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/layer_normalization/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/multi_head_attention/query/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/multi_head_attention/query/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/multi_head_attention/query/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/multi_head_attention/query/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/multi_head_attention/key/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention/key/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/multi_head_attention/key/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/multi_head_attention/key/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention/value/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention/value/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention/value/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention/value/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention/attention_output/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention/attention_output/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/m/multi_head_attention/attention_output/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/v/multi_head_attention/attention_output/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_1/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_1/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_1/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_1/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv1d/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1d/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_2/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_2/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_2/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_2/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_1/query/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_1/query/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_1/query/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_1/query/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_1/key/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_1/key/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention_1/key/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention_1/key/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_1/value/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_1/value/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_1/value/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_1/value/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/multi_head_attention_1/attention_output/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/multi_head_attention_1/attention_output/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention_1/attention_output/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention_1/attention_output/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_3/gamma2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_3/gamma2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_3/beta2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_3/beta2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_1/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_1/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_1/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_1/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_4/gamma2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_4/gamma2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_4/beta2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_4/beta2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_2/query/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_2/query/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_2/query/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_2/query/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_2/key/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_2/key/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention_2/key/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention_2/key/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_2/value/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_2/value/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_2/value/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_2/value/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/multi_head_attention_2/attention_output/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/multi_head_attention_2/attention_output/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention_2/attention_output/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention_2/attention_output/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_5/gamma2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_5/gamma2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_5/beta2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_5/beta2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_2/kernel2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_2/kernel2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_2/bias2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_2/bias2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_6/gamma2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_6/gamma2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_6/beta2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_6/beta2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_3/query/kernel2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_3/query/kernel2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_3/query/bias2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_3/query/bias2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_3/key/kernel2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_3/key/kernel2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention_3/key/bias2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention_3/key/bias2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_3/value/kernel2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_3/value/kernel2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_3/value/bias2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE(Adam/v/multi_head_attention_3/value/bias3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE5Adam/m/multi_head_attention_3/attention_output/kernel3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE5Adam/v/multi_head_attention_3/attention_output/kernel3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE3Adam/m/multi_head_attention_3/attention_output/bias3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE3Adam/v/multi_head_attention_3/attention_output/bias3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE"Adam/m/layer_normalization_7/gamma3optimizer/_variables/105/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE"Adam/v/layer_normalization_7/gamma3optimizer/_variables/106/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE!Adam/m/layer_normalization_7/beta3optimizer/_variables/107/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE!Adam/v/layer_normalization_7/beta3optimizer/_variables/108/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv1d_3/kernel3optimizer/_variables/109/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv1d_3/kernel3optimizer/_variables/110/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv1d_3/bias3optimizer/_variables/111/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv1d_3/bias3optimizer/_variables/112/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense/kernel3optimizer/_variables/113/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense/kernel3optimizer/_variables/114/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense/bias3optimizer/_variables/115/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/bias3optimizer/_variables/116/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_1/kernel3optimizer/_variables/117/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_1/kernel3optimizer/_variables/118/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/bias3optimizer/_variables/119/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/bias3optimizer/_variables/120/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�5
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betaconv1d_1/kernelconv1d_1/biaslayer_normalization_4/gammalayer_normalization_4/betalayer_normalization_5/gammalayer_normalization_5/betaconv1d_2/kernelconv1d_2/biaslayer_normalization_6/gammalayer_normalization_6/betalayer_normalization_7/gammalayer_normalization_7/betaconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/bias#multi_head_attention_3/query/kernel!multi_head_attention_3/query/bias!multi_head_attention_3/key/kernelmulti_head_attention_3/key/bias#multi_head_attention_3/value/kernel!multi_head_attention_3/value/bias.multi_head_attention_3/attention_output/kernel,multi_head_attention_3/attention_output/bias	iterationlearning_rate Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/beta(Adam/m/multi_head_attention/query/kernel(Adam/v/multi_head_attention/query/kernel&Adam/m/multi_head_attention/query/bias&Adam/v/multi_head_attention/query/bias&Adam/m/multi_head_attention/key/kernel&Adam/v/multi_head_attention/key/kernel$Adam/m/multi_head_attention/key/bias$Adam/v/multi_head_attention/key/bias(Adam/m/multi_head_attention/value/kernel(Adam/v/multi_head_attention/value/kernel&Adam/m/multi_head_attention/value/bias&Adam/v/multi_head_attention/value/bias3Adam/m/multi_head_attention/attention_output/kernel3Adam/v/multi_head_attention/attention_output/kernel1Adam/m/multi_head_attention/attention_output/bias1Adam/v/multi_head_attention/attention_output/bias"Adam/m/layer_normalization_1/gamma"Adam/v/layer_normalization_1/gamma!Adam/m/layer_normalization_1/beta!Adam/v/layer_normalization_1/betaAdam/m/conv1d/kernelAdam/v/conv1d/kernelAdam/m/conv1d/biasAdam/v/conv1d/bias"Adam/m/layer_normalization_2/gamma"Adam/v/layer_normalization_2/gamma!Adam/m/layer_normalization_2/beta!Adam/v/layer_normalization_2/beta*Adam/m/multi_head_attention_1/query/kernel*Adam/v/multi_head_attention_1/query/kernel(Adam/m/multi_head_attention_1/query/bias(Adam/v/multi_head_attention_1/query/bias(Adam/m/multi_head_attention_1/key/kernel(Adam/v/multi_head_attention_1/key/kernel&Adam/m/multi_head_attention_1/key/bias&Adam/v/multi_head_attention_1/key/bias*Adam/m/multi_head_attention_1/value/kernel*Adam/v/multi_head_attention_1/value/kernel(Adam/m/multi_head_attention_1/value/bias(Adam/v/multi_head_attention_1/value/bias5Adam/m/multi_head_attention_1/attention_output/kernel5Adam/v/multi_head_attention_1/attention_output/kernel3Adam/m/multi_head_attention_1/attention_output/bias3Adam/v/multi_head_attention_1/attention_output/bias"Adam/m/layer_normalization_3/gamma"Adam/v/layer_normalization_3/gamma!Adam/m/layer_normalization_3/beta!Adam/v/layer_normalization_3/betaAdam/m/conv1d_1/kernelAdam/v/conv1d_1/kernelAdam/m/conv1d_1/biasAdam/v/conv1d_1/bias"Adam/m/layer_normalization_4/gamma"Adam/v/layer_normalization_4/gamma!Adam/m/layer_normalization_4/beta!Adam/v/layer_normalization_4/beta*Adam/m/multi_head_attention_2/query/kernel*Adam/v/multi_head_attention_2/query/kernel(Adam/m/multi_head_attention_2/query/bias(Adam/v/multi_head_attention_2/query/bias(Adam/m/multi_head_attention_2/key/kernel(Adam/v/multi_head_attention_2/key/kernel&Adam/m/multi_head_attention_2/key/bias&Adam/v/multi_head_attention_2/key/bias*Adam/m/multi_head_attention_2/value/kernel*Adam/v/multi_head_attention_2/value/kernel(Adam/m/multi_head_attention_2/value/bias(Adam/v/multi_head_attention_2/value/bias5Adam/m/multi_head_attention_2/attention_output/kernel5Adam/v/multi_head_attention_2/attention_output/kernel3Adam/m/multi_head_attention_2/attention_output/bias3Adam/v/multi_head_attention_2/attention_output/bias"Adam/m/layer_normalization_5/gamma"Adam/v/layer_normalization_5/gamma!Adam/m/layer_normalization_5/beta!Adam/v/layer_normalization_5/betaAdam/m/conv1d_2/kernelAdam/v/conv1d_2/kernelAdam/m/conv1d_2/biasAdam/v/conv1d_2/bias"Adam/m/layer_normalization_6/gamma"Adam/v/layer_normalization_6/gamma!Adam/m/layer_normalization_6/beta!Adam/v/layer_normalization_6/beta*Adam/m/multi_head_attention_3/query/kernel*Adam/v/multi_head_attention_3/query/kernel(Adam/m/multi_head_attention_3/query/bias(Adam/v/multi_head_attention_3/query/bias(Adam/m/multi_head_attention_3/key/kernel(Adam/v/multi_head_attention_3/key/kernel&Adam/m/multi_head_attention_3/key/bias&Adam/v/multi_head_attention_3/key/bias*Adam/m/multi_head_attention_3/value/kernel*Adam/v/multi_head_attention_3/value/kernel(Adam/m/multi_head_attention_3/value/bias(Adam/v/multi_head_attention_3/value/bias5Adam/m/multi_head_attention_3/attention_output/kernel5Adam/v/multi_head_attention_3/attention_output/kernel3Adam/m/multi_head_attention_3/attention_output/bias3Adam/v/multi_head_attention_3/attention_output/bias"Adam/m/layer_normalization_7/gamma"Adam/v/layer_normalization_7/gamma!Adam/m/layer_normalization_7/beta!Adam/v/layer_normalization_7/betaAdam/m/conv1d_3/kernelAdam/v/conv1d_3/kernelAdam/m/conv1d_3/biasAdam/v/conv1d_3/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcountConst*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_240969
�5
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betaconv1d_1/kernelconv1d_1/biaslayer_normalization_4/gammalayer_normalization_4/betalayer_normalization_5/gammalayer_normalization_5/betaconv1d_2/kernelconv1d_2/biaslayer_normalization_6/gammalayer_normalization_6/betalayer_normalization_7/gammalayer_normalization_7/betaconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/bias#multi_head_attention_3/query/kernel!multi_head_attention_3/query/bias!multi_head_attention_3/key/kernelmulti_head_attention_3/key/bias#multi_head_attention_3/value/kernel!multi_head_attention_3/value/bias.multi_head_attention_3/attention_output/kernel,multi_head_attention_3/attention_output/bias	iterationlearning_rate Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/beta(Adam/m/multi_head_attention/query/kernel(Adam/v/multi_head_attention/query/kernel&Adam/m/multi_head_attention/query/bias&Adam/v/multi_head_attention/query/bias&Adam/m/multi_head_attention/key/kernel&Adam/v/multi_head_attention/key/kernel$Adam/m/multi_head_attention/key/bias$Adam/v/multi_head_attention/key/bias(Adam/m/multi_head_attention/value/kernel(Adam/v/multi_head_attention/value/kernel&Adam/m/multi_head_attention/value/bias&Adam/v/multi_head_attention/value/bias3Adam/m/multi_head_attention/attention_output/kernel3Adam/v/multi_head_attention/attention_output/kernel1Adam/m/multi_head_attention/attention_output/bias1Adam/v/multi_head_attention/attention_output/bias"Adam/m/layer_normalization_1/gamma"Adam/v/layer_normalization_1/gamma!Adam/m/layer_normalization_1/beta!Adam/v/layer_normalization_1/betaAdam/m/conv1d/kernelAdam/v/conv1d/kernelAdam/m/conv1d/biasAdam/v/conv1d/bias"Adam/m/layer_normalization_2/gamma"Adam/v/layer_normalization_2/gamma!Adam/m/layer_normalization_2/beta!Adam/v/layer_normalization_2/beta*Adam/m/multi_head_attention_1/query/kernel*Adam/v/multi_head_attention_1/query/kernel(Adam/m/multi_head_attention_1/query/bias(Adam/v/multi_head_attention_1/query/bias(Adam/m/multi_head_attention_1/key/kernel(Adam/v/multi_head_attention_1/key/kernel&Adam/m/multi_head_attention_1/key/bias&Adam/v/multi_head_attention_1/key/bias*Adam/m/multi_head_attention_1/value/kernel*Adam/v/multi_head_attention_1/value/kernel(Adam/m/multi_head_attention_1/value/bias(Adam/v/multi_head_attention_1/value/bias5Adam/m/multi_head_attention_1/attention_output/kernel5Adam/v/multi_head_attention_1/attention_output/kernel3Adam/m/multi_head_attention_1/attention_output/bias3Adam/v/multi_head_attention_1/attention_output/bias"Adam/m/layer_normalization_3/gamma"Adam/v/layer_normalization_3/gamma!Adam/m/layer_normalization_3/beta!Adam/v/layer_normalization_3/betaAdam/m/conv1d_1/kernelAdam/v/conv1d_1/kernelAdam/m/conv1d_1/biasAdam/v/conv1d_1/bias"Adam/m/layer_normalization_4/gamma"Adam/v/layer_normalization_4/gamma!Adam/m/layer_normalization_4/beta!Adam/v/layer_normalization_4/beta*Adam/m/multi_head_attention_2/query/kernel*Adam/v/multi_head_attention_2/query/kernel(Adam/m/multi_head_attention_2/query/bias(Adam/v/multi_head_attention_2/query/bias(Adam/m/multi_head_attention_2/key/kernel(Adam/v/multi_head_attention_2/key/kernel&Adam/m/multi_head_attention_2/key/bias&Adam/v/multi_head_attention_2/key/bias*Adam/m/multi_head_attention_2/value/kernel*Adam/v/multi_head_attention_2/value/kernel(Adam/m/multi_head_attention_2/value/bias(Adam/v/multi_head_attention_2/value/bias5Adam/m/multi_head_attention_2/attention_output/kernel5Adam/v/multi_head_attention_2/attention_output/kernel3Adam/m/multi_head_attention_2/attention_output/bias3Adam/v/multi_head_attention_2/attention_output/bias"Adam/m/layer_normalization_5/gamma"Adam/v/layer_normalization_5/gamma!Adam/m/layer_normalization_5/beta!Adam/v/layer_normalization_5/betaAdam/m/conv1d_2/kernelAdam/v/conv1d_2/kernelAdam/m/conv1d_2/biasAdam/v/conv1d_2/bias"Adam/m/layer_normalization_6/gamma"Adam/v/layer_normalization_6/gamma!Adam/m/layer_normalization_6/beta!Adam/v/layer_normalization_6/beta*Adam/m/multi_head_attention_3/query/kernel*Adam/v/multi_head_attention_3/query/kernel(Adam/m/multi_head_attention_3/query/bias(Adam/v/multi_head_attention_3/query/bias(Adam/m/multi_head_attention_3/key/kernel(Adam/v/multi_head_attention_3/key/kernel&Adam/m/multi_head_attention_3/key/bias&Adam/v/multi_head_attention_3/key/bias*Adam/m/multi_head_attention_3/value/kernel*Adam/v/multi_head_attention_3/value/kernel(Adam/m/multi_head_attention_3/value/bias(Adam/v/multi_head_attention_3/value/bias5Adam/m/multi_head_attention_3/attention_output/kernel5Adam/v/multi_head_attention_3/attention_output/kernel3Adam/m/multi_head_attention_3/attention_output/bias3Adam/v/multi_head_attention_3/attention_output/bias"Adam/m/layer_normalization_7/gamma"Adam/v/layer_normalization_7/gamma!Adam/m/layer_normalization_7/beta!Adam/v/layer_normalization_7/betaAdam/m/conv1d_3/kernelAdam/v/conv1d_3/kernelAdam/m/conv1d_3/biasAdam/v/conv1d_3/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcount*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_241536��)
�4
�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239612	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_239737

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_4_layer_call_fn_239245

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_237272s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239241:&"
 
_user_specified_name239239:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_1_layer_call_fn_239030	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237158s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239026:&"
 
_user_specified_name239024:&"
 
_user_specified_name239022:&"
 
_user_specified_name239020:&"
 
_user_specified_name239018:&"
 
_user_specified_name239016:&"
 
_user_specified_name239014:&"
 
_user_specified_name239012:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
7__inference_multi_head_attention_2_layer_call_fn_239311	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237811s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239307:&"
 
_user_specified_name239305:&"
 
_user_specified_name239303:&"
 
_user_specified_name239301:&"
 
_user_specified_name239299:&"
 
_user_specified_name239297:&"
 
_user_specified_name239295:&"
 
_user_specified_name239293:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_239831

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_237272

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239647	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_239742

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_1_layer_call_fn_238891

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_237043s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name238887:&"
 
_user_specified_name238885:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_237619
input_1(
layer_normalization_236951:(
layer_normalization_236953:1
multi_head_attention_236998:@-
multi_head_attention_237000:@1
multi_head_attention_237002:@-
multi_head_attention_237004:@1
multi_head_attention_237006:@-
multi_head_attention_237008:@1
multi_head_attention_237010:@)
multi_head_attention_237012:*
layer_normalization_1_237044:*
layer_normalization_1_237046:#
conv1d_237065:
conv1d_237067:*
layer_normalization_2_237112:*
layer_normalization_2_237114:3
multi_head_attention_1_237159:@/
multi_head_attention_1_237161:@3
multi_head_attention_1_237163:@/
multi_head_attention_1_237165:@3
multi_head_attention_1_237167:@/
multi_head_attention_1_237169:@3
multi_head_attention_1_237171:@+
multi_head_attention_1_237173:*
layer_normalization_3_237205:*
layer_normalization_3_237207:%
conv1d_1_237226:
conv1d_1_237228:*
layer_normalization_4_237273:*
layer_normalization_4_237275:3
multi_head_attention_2_237320:@/
multi_head_attention_2_237322:@3
multi_head_attention_2_237324:@/
multi_head_attention_2_237326:@3
multi_head_attention_2_237328:@/
multi_head_attention_2_237330:@3
multi_head_attention_2_237332:@+
multi_head_attention_2_237334:*
layer_normalization_5_237366:*
layer_normalization_5_237368:%
conv1d_2_237387:
conv1d_2_237389:*
layer_normalization_6_237434:*
layer_normalization_6_237436:3
multi_head_attention_3_237481:@/
multi_head_attention_3_237483:@3
multi_head_attention_3_237485:@/
multi_head_attention_3_237487:@3
multi_head_attention_3_237489:@/
multi_head_attention_3_237491:@3
multi_head_attention_3_237493:@+
multi_head_attention_3_237495:*
layer_normalization_7_237527:*
layer_normalization_7_237529:%
conv1d_3_237548:
conv1d_3_237550:
dense_237585:	�
dense_237587:	�!
dense_1_237613:	�
dense_1_237615:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�.multi_head_attention_2/StatefulPartitionedCall�.multi_head_attention_3/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_236951layer_normalization_236953*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_236950�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_236998multi_head_attention_237000multi_head_attention_237002multi_head_attention_237004multi_head_attention_237006multi_head_attention_237008multi_head_attention_237010multi_head_attention_237012*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_236997�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_237020�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_237044layer_normalization_1_237046*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_237043�
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_237065conv1d_237067*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_237064�
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_237081�
add_1/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_237088�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_237112layer_normalization_2_237114*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_237111�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_237159multi_head_attention_1_237161multi_head_attention_1_237163multi_head_attention_1_237165multi_head_attention_1_237167multi_head_attention_1_237169multi_head_attention_1_237171multi_head_attention_1_237173*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237158�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_237181�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_237205layer_normalization_3_237207*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_237204�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_1_237226conv1d_1_237228*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_237225�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_237242�
add_3/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_237249�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_237273layer_normalization_4_237275*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_237272�
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_237320multi_head_attention_2_237322multi_head_attention_2_237324multi_head_attention_2_237326multi_head_attention_2_237328multi_head_attention_2_237330multi_head_attention_2_237332multi_head_attention_2_237334*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237319�
add_4/PartitionedCallPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_237342�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0layer_normalization_5_237366layer_normalization_5_237368*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_237365�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_2_237387conv1d_2_237389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_237386�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_237403�
add_5/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_237410�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0layer_normalization_6_237434layer_normalization_6_237436*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_237433�
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_237481multi_head_attention_3_237483multi_head_attention_3_237485multi_head_attention_3_237487multi_head_attention_3_237489multi_head_attention_3_237491multi_head_attention_3_237493multi_head_attention_3_237495*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237480�
add_6/PartitionedCallPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_237503�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0layer_normalization_7_237527layer_normalization_7_237529*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_237526�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_3_237548conv1d_3_237550*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_237547�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_237564�
add_7/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0add_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_237571�
(global_average_pooling1d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_236921�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_237585dense_237587*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_237584�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_237601�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_1_237613dense_1_237615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_237612w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:&<"
 
_user_specified_name237615:&;"
 
_user_specified_name237613:&:"
 
_user_specified_name237587:&9"
 
_user_specified_name237585:&8"
 
_user_specified_name237550:&7"
 
_user_specified_name237548:&6"
 
_user_specified_name237529:&5"
 
_user_specified_name237527:&4"
 
_user_specified_name237495:&3"
 
_user_specified_name237493:&2"
 
_user_specified_name237491:&1"
 
_user_specified_name237489:&0"
 
_user_specified_name237487:&/"
 
_user_specified_name237485:&."
 
_user_specified_name237483:&-"
 
_user_specified_name237481:&,"
 
_user_specified_name237436:&+"
 
_user_specified_name237434:&*"
 
_user_specified_name237389:&)"
 
_user_specified_name237387:&("
 
_user_specified_name237368:&'"
 
_user_specified_name237366:&&"
 
_user_specified_name237334:&%"
 
_user_specified_name237332:&$"
 
_user_specified_name237330:&#"
 
_user_specified_name237328:&""
 
_user_specified_name237326:&!"
 
_user_specified_name237324:& "
 
_user_specified_name237322:&"
 
_user_specified_name237320:&"
 
_user_specified_name237275:&"
 
_user_specified_name237273:&"
 
_user_specified_name237228:&"
 
_user_specified_name237226:&"
 
_user_specified_name237207:&"
 
_user_specified_name237205:&"
 
_user_specified_name237173:&"
 
_user_specified_name237171:&"
 
_user_specified_name237169:&"
 
_user_specified_name237167:&"
 
_user_specified_name237165:&"
 
_user_specified_name237163:&"
 
_user_specified_name237161:&"
 
_user_specified_name237159:&"
 
_user_specified_name237114:&"
 
_user_specified_name237112:&"
 
_user_specified_name237067:&"
 
_user_specified_name237065:&"
 
_user_specified_name237046:&"
 
_user_specified_name237044:&
"
 
_user_specified_name237012:&	"
 
_user_specified_name237010:&"
 
_user_specified_name237008:&"
 
_user_specified_name237006:&"
 
_user_specified_name237004:&"
 
_user_specified_name237002:&"
 
_user_specified_name237000:&"
 
_user_specified_name236998:&"
 
_user_specified_name236953:&"
 
_user_specified_name236951:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
6__inference_layer_normalization_2_layer_call_fn_238986

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_237111s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name238982:&"
 
_user_specified_name238980:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_239774

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_237584p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239770:&"
 
_user_specified_name239768:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
A__inference_add_7_layer_call_and_return_conditional_losses_237571

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv1d_3_layer_call_fn_239699

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_237547s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239695:&"
 
_user_specified_name239693:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_237111

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_239785

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_239141
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_238913

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_239821

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_237612o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239817:&"
 
_user_specified_name239815:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
R
&__inference_add_6_layer_call_fn_239653
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_237503d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�,
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239129	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_239765

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_1_layer_call_fn_239052	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237736s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239048:&"
 
_user_specified_name239046:&"
 
_user_specified_name239044:&"
 
_user_specified_name239042:&"
 
_user_specified_name239040:&"
 
_user_specified_name239038:&"
 
_user_specified_name239036:&"
 
_user_specified_name239034:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_238965

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_2_layer_call_fn_239289	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237319s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239285:&"
 
_user_specified_name239283:&"
 
_user_specified_name239281:&"
 
_user_specified_name239279:&"
 
_user_specified_name239277:&"
 
_user_specified_name239275:&"
 
_user_specified_name239273:&"
 
_user_specified_name239271:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
B__inference_conv1d_layer_call_and_return_conditional_losses_238938

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv1d_2_layer_call_fn_239440

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_237386s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239436:&"
 
_user_specified_name239434:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
?__inference_add_layer_call_and_return_conditional_losses_238882
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
R
&__inference_add_2_layer_call_fn_239135
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_237181d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_239267

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_237919

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
U
9__inference_global_average_pooling1d_layer_call_fn_239759

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_236921i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
B__inference_conv1d_layer_call_and_return_conditional_losses_237064

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_2_layer_call_fn_239461

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_237403s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239353	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_239478

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_236997	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�.
�
$__inference_signature_wrapper_238718
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15:@

unknown_16:@ 

unknown_17:@

unknown_18:@ 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28: 

unknown_29:@

unknown_30:@ 

unknown_31:@

unknown_32:@ 

unknown_33:@

unknown_34:@ 

unknown_35:@

unknown_36:

unknown_37:

unknown_38: 

unknown_39:

unknown_40:

unknown_41:

unknown_42: 

unknown_43:@

unknown_44:@ 

unknown_45:@

unknown_46:@ 

unknown_47:@

unknown_48:@ 

unknown_49:@

unknown_50:

unknown_51:

unknown_52: 

unknown_53:

unknown_54:

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_236915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&<"
 
_user_specified_name238714:&;"
 
_user_specified_name238712:&:"
 
_user_specified_name238710:&9"
 
_user_specified_name238708:&8"
 
_user_specified_name238706:&7"
 
_user_specified_name238704:&6"
 
_user_specified_name238702:&5"
 
_user_specified_name238700:&4"
 
_user_specified_name238698:&3"
 
_user_specified_name238696:&2"
 
_user_specified_name238694:&1"
 
_user_specified_name238692:&0"
 
_user_specified_name238690:&/"
 
_user_specified_name238688:&."
 
_user_specified_name238686:&-"
 
_user_specified_name238684:&,"
 
_user_specified_name238682:&+"
 
_user_specified_name238680:&*"
 
_user_specified_name238678:&)"
 
_user_specified_name238676:&("
 
_user_specified_name238674:&'"
 
_user_specified_name238672:&&"
 
_user_specified_name238670:&%"
 
_user_specified_name238668:&$"
 
_user_specified_name238666:&#"
 
_user_specified_name238664:&""
 
_user_specified_name238662:&!"
 
_user_specified_name238660:& "
 
_user_specified_name238658:&"
 
_user_specified_name238656:&"
 
_user_specified_name238654:&"
 
_user_specified_name238652:&"
 
_user_specified_name238650:&"
 
_user_specified_name238648:&"
 
_user_specified_name238646:&"
 
_user_specified_name238644:&"
 
_user_specified_name238642:&"
 
_user_specified_name238640:&"
 
_user_specified_name238638:&"
 
_user_specified_name238636:&"
 
_user_specified_name238634:&"
 
_user_specified_name238632:&"
 
_user_specified_name238630:&"
 
_user_specified_name238628:&"
 
_user_specified_name238626:&"
 
_user_specified_name238624:&"
 
_user_specified_name238622:&"
 
_user_specified_name238620:&"
 
_user_specified_name238618:&"
 
_user_specified_name238616:&
"
 
_user_specified_name238614:&	"
 
_user_specified_name238612:&"
 
_user_specified_name238610:&"
 
_user_specified_name238608:&"
 
_user_specified_name238606:&"
 
_user_specified_name238604:&"
 
_user_specified_name238602:&"
 
_user_specified_name238600:&"
 
_user_specified_name238598:&"
 
_user_specified_name238596:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_237932

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_3_layer_call_and_return_conditional_losses_237547

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
m
A__inference_add_6_layer_call_and_return_conditional_losses_239659
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_238960

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_237526

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238870	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_238977
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
��
��
"__inference__traced_restore_241536
file_prefix8
*assignvariableop_layer_normalization_gamma:9
+assignvariableop_1_layer_normalization_beta:<
.assignvariableop_2_layer_normalization_1_gamma:;
-assignvariableop_3_layer_normalization_1_beta:6
 assignvariableop_4_conv1d_kernel:,
assignvariableop_5_conv1d_bias:<
.assignvariableop_6_layer_normalization_2_gamma:;
-assignvariableop_7_layer_normalization_2_beta:<
.assignvariableop_8_layer_normalization_3_gamma:;
-assignvariableop_9_layer_normalization_3_beta:9
#assignvariableop_10_conv1d_1_kernel:/
!assignvariableop_11_conv1d_1_bias:=
/assignvariableop_12_layer_normalization_4_gamma:<
.assignvariableop_13_layer_normalization_4_beta:=
/assignvariableop_14_layer_normalization_5_gamma:<
.assignvariableop_15_layer_normalization_5_beta:9
#assignvariableop_16_conv1d_2_kernel:/
!assignvariableop_17_conv1d_2_bias:=
/assignvariableop_18_layer_normalization_6_gamma:<
.assignvariableop_19_layer_normalization_6_beta:=
/assignvariableop_20_layer_normalization_7_gamma:<
.assignvariableop_21_layer_normalization_7_beta:9
#assignvariableop_22_conv1d_3_kernel:/
!assignvariableop_23_conv1d_3_bias:3
 assignvariableop_24_dense_kernel:	�-
assignvariableop_25_dense_bias:	�5
"assignvariableop_26_dense_1_kernel:	�.
 assignvariableop_27_dense_1_bias:K
5assignvariableop_28_multi_head_attention_query_kernel:@E
3assignvariableop_29_multi_head_attention_query_bias:@I
3assignvariableop_30_multi_head_attention_key_kernel:@C
1assignvariableop_31_multi_head_attention_key_bias:@K
5assignvariableop_32_multi_head_attention_value_kernel:@E
3assignvariableop_33_multi_head_attention_value_bias:@V
@assignvariableop_34_multi_head_attention_attention_output_kernel:@L
>assignvariableop_35_multi_head_attention_attention_output_bias:M
7assignvariableop_36_multi_head_attention_1_query_kernel:@G
5assignvariableop_37_multi_head_attention_1_query_bias:@K
5assignvariableop_38_multi_head_attention_1_key_kernel:@E
3assignvariableop_39_multi_head_attention_1_key_bias:@M
7assignvariableop_40_multi_head_attention_1_value_kernel:@G
5assignvariableop_41_multi_head_attention_1_value_bias:@X
Bassignvariableop_42_multi_head_attention_1_attention_output_kernel:@N
@assignvariableop_43_multi_head_attention_1_attention_output_bias:M
7assignvariableop_44_multi_head_attention_2_query_kernel:@G
5assignvariableop_45_multi_head_attention_2_query_bias:@K
5assignvariableop_46_multi_head_attention_2_key_kernel:@E
3assignvariableop_47_multi_head_attention_2_key_bias:@M
7assignvariableop_48_multi_head_attention_2_value_kernel:@G
5assignvariableop_49_multi_head_attention_2_value_bias:@X
Bassignvariableop_50_multi_head_attention_2_attention_output_kernel:@N
@assignvariableop_51_multi_head_attention_2_attention_output_bias:M
7assignvariableop_52_multi_head_attention_3_query_kernel:@G
5assignvariableop_53_multi_head_attention_3_query_bias:@K
5assignvariableop_54_multi_head_attention_3_key_kernel:@E
3assignvariableop_55_multi_head_attention_3_key_bias:@M
7assignvariableop_56_multi_head_attention_3_value_kernel:@G
5assignvariableop_57_multi_head_attention_3_value_bias:@X
Bassignvariableop_58_multi_head_attention_3_attention_output_kernel:@N
@assignvariableop_59_multi_head_attention_3_attention_output_bias:'
assignvariableop_60_iteration:	 +
!assignvariableop_61_learning_rate: B
4assignvariableop_62_adam_m_layer_normalization_gamma:B
4assignvariableop_63_adam_v_layer_normalization_gamma:A
3assignvariableop_64_adam_m_layer_normalization_beta:A
3assignvariableop_65_adam_v_layer_normalization_beta:R
<assignvariableop_66_adam_m_multi_head_attention_query_kernel:@R
<assignvariableop_67_adam_v_multi_head_attention_query_kernel:@L
:assignvariableop_68_adam_m_multi_head_attention_query_bias:@L
:assignvariableop_69_adam_v_multi_head_attention_query_bias:@P
:assignvariableop_70_adam_m_multi_head_attention_key_kernel:@P
:assignvariableop_71_adam_v_multi_head_attention_key_kernel:@J
8assignvariableop_72_adam_m_multi_head_attention_key_bias:@J
8assignvariableop_73_adam_v_multi_head_attention_key_bias:@R
<assignvariableop_74_adam_m_multi_head_attention_value_kernel:@R
<assignvariableop_75_adam_v_multi_head_attention_value_kernel:@L
:assignvariableop_76_adam_m_multi_head_attention_value_bias:@L
:assignvariableop_77_adam_v_multi_head_attention_value_bias:@]
Gassignvariableop_78_adam_m_multi_head_attention_attention_output_kernel:@]
Gassignvariableop_79_adam_v_multi_head_attention_attention_output_kernel:@S
Eassignvariableop_80_adam_m_multi_head_attention_attention_output_bias:S
Eassignvariableop_81_adam_v_multi_head_attention_attention_output_bias:D
6assignvariableop_82_adam_m_layer_normalization_1_gamma:D
6assignvariableop_83_adam_v_layer_normalization_1_gamma:C
5assignvariableop_84_adam_m_layer_normalization_1_beta:C
5assignvariableop_85_adam_v_layer_normalization_1_beta:>
(assignvariableop_86_adam_m_conv1d_kernel:>
(assignvariableop_87_adam_v_conv1d_kernel:4
&assignvariableop_88_adam_m_conv1d_bias:4
&assignvariableop_89_adam_v_conv1d_bias:D
6assignvariableop_90_adam_m_layer_normalization_2_gamma:D
6assignvariableop_91_adam_v_layer_normalization_2_gamma:C
5assignvariableop_92_adam_m_layer_normalization_2_beta:C
5assignvariableop_93_adam_v_layer_normalization_2_beta:T
>assignvariableop_94_adam_m_multi_head_attention_1_query_kernel:@T
>assignvariableop_95_adam_v_multi_head_attention_1_query_kernel:@N
<assignvariableop_96_adam_m_multi_head_attention_1_query_bias:@N
<assignvariableop_97_adam_v_multi_head_attention_1_query_bias:@R
<assignvariableop_98_adam_m_multi_head_attention_1_key_kernel:@R
<assignvariableop_99_adam_v_multi_head_attention_1_key_kernel:@M
;assignvariableop_100_adam_m_multi_head_attention_1_key_bias:@M
;assignvariableop_101_adam_v_multi_head_attention_1_key_bias:@U
?assignvariableop_102_adam_m_multi_head_attention_1_value_kernel:@U
?assignvariableop_103_adam_v_multi_head_attention_1_value_kernel:@O
=assignvariableop_104_adam_m_multi_head_attention_1_value_bias:@O
=assignvariableop_105_adam_v_multi_head_attention_1_value_bias:@`
Jassignvariableop_106_adam_m_multi_head_attention_1_attention_output_kernel:@`
Jassignvariableop_107_adam_v_multi_head_attention_1_attention_output_kernel:@V
Hassignvariableop_108_adam_m_multi_head_attention_1_attention_output_bias:V
Hassignvariableop_109_adam_v_multi_head_attention_1_attention_output_bias:E
7assignvariableop_110_adam_m_layer_normalization_3_gamma:E
7assignvariableop_111_adam_v_layer_normalization_3_gamma:D
6assignvariableop_112_adam_m_layer_normalization_3_beta:D
6assignvariableop_113_adam_v_layer_normalization_3_beta:A
+assignvariableop_114_adam_m_conv1d_1_kernel:A
+assignvariableop_115_adam_v_conv1d_1_kernel:7
)assignvariableop_116_adam_m_conv1d_1_bias:7
)assignvariableop_117_adam_v_conv1d_1_bias:E
7assignvariableop_118_adam_m_layer_normalization_4_gamma:E
7assignvariableop_119_adam_v_layer_normalization_4_gamma:D
6assignvariableop_120_adam_m_layer_normalization_4_beta:D
6assignvariableop_121_adam_v_layer_normalization_4_beta:U
?assignvariableop_122_adam_m_multi_head_attention_2_query_kernel:@U
?assignvariableop_123_adam_v_multi_head_attention_2_query_kernel:@O
=assignvariableop_124_adam_m_multi_head_attention_2_query_bias:@O
=assignvariableop_125_adam_v_multi_head_attention_2_query_bias:@S
=assignvariableop_126_adam_m_multi_head_attention_2_key_kernel:@S
=assignvariableop_127_adam_v_multi_head_attention_2_key_kernel:@M
;assignvariableop_128_adam_m_multi_head_attention_2_key_bias:@M
;assignvariableop_129_adam_v_multi_head_attention_2_key_bias:@U
?assignvariableop_130_adam_m_multi_head_attention_2_value_kernel:@U
?assignvariableop_131_adam_v_multi_head_attention_2_value_kernel:@O
=assignvariableop_132_adam_m_multi_head_attention_2_value_bias:@O
=assignvariableop_133_adam_v_multi_head_attention_2_value_bias:@`
Jassignvariableop_134_adam_m_multi_head_attention_2_attention_output_kernel:@`
Jassignvariableop_135_adam_v_multi_head_attention_2_attention_output_kernel:@V
Hassignvariableop_136_adam_m_multi_head_attention_2_attention_output_bias:V
Hassignvariableop_137_adam_v_multi_head_attention_2_attention_output_bias:E
7assignvariableop_138_adam_m_layer_normalization_5_gamma:E
7assignvariableop_139_adam_v_layer_normalization_5_gamma:D
6assignvariableop_140_adam_m_layer_normalization_5_beta:D
6assignvariableop_141_adam_v_layer_normalization_5_beta:A
+assignvariableop_142_adam_m_conv1d_2_kernel:A
+assignvariableop_143_adam_v_conv1d_2_kernel:7
)assignvariableop_144_adam_m_conv1d_2_bias:7
)assignvariableop_145_adam_v_conv1d_2_bias:E
7assignvariableop_146_adam_m_layer_normalization_6_gamma:E
7assignvariableop_147_adam_v_layer_normalization_6_gamma:D
6assignvariableop_148_adam_m_layer_normalization_6_beta:D
6assignvariableop_149_adam_v_layer_normalization_6_beta:U
?assignvariableop_150_adam_m_multi_head_attention_3_query_kernel:@U
?assignvariableop_151_adam_v_multi_head_attention_3_query_kernel:@O
=assignvariableop_152_adam_m_multi_head_attention_3_query_bias:@O
=assignvariableop_153_adam_v_multi_head_attention_3_query_bias:@S
=assignvariableop_154_adam_m_multi_head_attention_3_key_kernel:@S
=assignvariableop_155_adam_v_multi_head_attention_3_key_kernel:@M
;assignvariableop_156_adam_m_multi_head_attention_3_key_bias:@M
;assignvariableop_157_adam_v_multi_head_attention_3_key_bias:@U
?assignvariableop_158_adam_m_multi_head_attention_3_value_kernel:@U
?assignvariableop_159_adam_v_multi_head_attention_3_value_kernel:@O
=assignvariableop_160_adam_m_multi_head_attention_3_value_bias:@O
=assignvariableop_161_adam_v_multi_head_attention_3_value_bias:@`
Jassignvariableop_162_adam_m_multi_head_attention_3_attention_output_kernel:@`
Jassignvariableop_163_adam_v_multi_head_attention_3_attention_output_kernel:@V
Hassignvariableop_164_adam_m_multi_head_attention_3_attention_output_bias:V
Hassignvariableop_165_adam_v_multi_head_attention_3_attention_output_bias:E
7assignvariableop_166_adam_m_layer_normalization_7_gamma:E
7assignvariableop_167_adam_v_layer_normalization_7_gamma:D
6assignvariableop_168_adam_m_layer_normalization_7_beta:D
6assignvariableop_169_adam_v_layer_normalization_7_beta:A
+assignvariableop_170_adam_m_conv1d_3_kernel:A
+assignvariableop_171_adam_v_conv1d_3_kernel:7
)assignvariableop_172_adam_m_conv1d_3_bias:7
)assignvariableop_173_adam_v_conv1d_3_bias:;
(assignvariableop_174_adam_m_dense_kernel:	�;
(assignvariableop_175_adam_v_dense_kernel:	�5
&assignvariableop_176_adam_m_dense_bias:	�5
&assignvariableop_177_adam_v_dense_bias:	�=
*assignvariableop_178_adam_m_dense_1_kernel:	�=
*assignvariableop_179_adam_v_dense_1_kernel:	�6
(assignvariableop_180_adam_m_dense_1_bias:6
(assignvariableop_181_adam_v_dense_1_bias:&
assignvariableop_182_total_1: &
assignvariableop_183_count_1: $
assignvariableop_184_total: $
assignvariableop_185_count: 
identity_187��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_163�AssignVariableOp_164�AssignVariableOp_165�AssignVariableOp_166�AssignVariableOp_167�AssignVariableOp_168�AssignVariableOp_169�AssignVariableOp_17�AssignVariableOp_170�AssignVariableOp_171�AssignVariableOp_172�AssignVariableOp_173�AssignVariableOp_174�AssignVariableOp_175�AssignVariableOp_176�AssignVariableOp_177�AssignVariableOp_178�AssignVariableOp_179�AssignVariableOp_18�AssignVariableOp_180�AssignVariableOp_181�AssignVariableOp_182�AssignVariableOp_183�AssignVariableOp_184�AssignVariableOp_185�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�J
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�J
value�IB�I�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/105/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/106/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/107/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/108/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/109/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/110/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/111/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/112/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/113/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/114/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/115/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/116/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/117/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/118/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/119/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/120/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1d_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_2_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_2_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_3_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv1d_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_4_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp.assignvariableop_13_layer_normalization_4_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_layer_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_layer_normalization_5_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv1d_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv1d_2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_layer_normalization_6_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_layer_normalization_6_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_layer_normalization_7_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_layer_normalization_7_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv1d_3_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv1d_3_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_dense_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_1_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_query_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_query_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_multi_head_attention_key_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp1assignvariableop_31_multi_head_attention_key_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_multi_head_attention_value_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp3assignvariableop_33_multi_head_attention_value_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp@assignvariableop_34_multi_head_attention_attention_output_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp>assignvariableop_35_multi_head_attention_attention_output_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp7assignvariableop_36_multi_head_attention_1_query_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_multi_head_attention_1_query_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_multi_head_attention_1_key_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp3assignvariableop_39_multi_head_attention_1_key_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp7assignvariableop_40_multi_head_attention_1_value_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_multi_head_attention_1_value_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpBassignvariableop_42_multi_head_attention_1_attention_output_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp@assignvariableop_43_multi_head_attention_1_attention_output_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_multi_head_attention_2_query_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp5assignvariableop_45_multi_head_attention_2_query_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_multi_head_attention_2_key_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp3assignvariableop_47_multi_head_attention_2_key_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp7assignvariableop_48_multi_head_attention_2_value_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp5assignvariableop_49_multi_head_attention_2_value_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpBassignvariableop_50_multi_head_attention_2_attention_output_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp@assignvariableop_51_multi_head_attention_2_attention_output_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp7assignvariableop_52_multi_head_attention_3_query_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp5assignvariableop_53_multi_head_attention_3_query_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp5assignvariableop_54_multi_head_attention_3_key_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp3assignvariableop_55_multi_head_attention_3_key_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp7assignvariableop_56_multi_head_attention_3_value_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp5assignvariableop_57_multi_head_attention_3_value_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpBassignvariableop_58_multi_head_attention_3_attention_output_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp@assignvariableop_59_multi_head_attention_3_attention_output_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_iterationIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp!assignvariableop_61_learning_rateIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_m_layer_normalization_gammaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_v_layer_normalization_gammaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp3assignvariableop_64_adam_m_layer_normalization_betaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp3assignvariableop_65_adam_v_layer_normalization_betaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp<assignvariableop_66_adam_m_multi_head_attention_query_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp<assignvariableop_67_adam_v_multi_head_attention_query_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp:assignvariableop_68_adam_m_multi_head_attention_query_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp:assignvariableop_69_adam_v_multi_head_attention_query_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp:assignvariableop_70_adam_m_multi_head_attention_key_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp:assignvariableop_71_adam_v_multi_head_attention_key_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp8assignvariableop_72_adam_m_multi_head_attention_key_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_v_multi_head_attention_key_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp<assignvariableop_74_adam_m_multi_head_attention_value_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp<assignvariableop_75_adam_v_multi_head_attention_value_kernelIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp:assignvariableop_76_adam_m_multi_head_attention_value_biasIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp:assignvariableop_77_adam_v_multi_head_attention_value_biasIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpGassignvariableop_78_adam_m_multi_head_attention_attention_output_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpGassignvariableop_79_adam_v_multi_head_attention_attention_output_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpEassignvariableop_80_adam_m_multi_head_attention_attention_output_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpEassignvariableop_81_adam_v_multi_head_attention_attention_output_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_m_layer_normalization_1_gammaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_v_layer_normalization_1_gammaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adam_m_layer_normalization_1_betaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_v_layer_normalization_1_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_m_conv1d_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_v_conv1d_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_m_conv1d_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp&assignvariableop_89_adam_v_conv1d_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_m_layer_normalization_2_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adam_v_layer_normalization_2_gammaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp5assignvariableop_92_adam_m_layer_normalization_2_betaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp5assignvariableop_93_adam_v_layer_normalization_2_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp>assignvariableop_94_adam_m_multi_head_attention_1_query_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp>assignvariableop_95_adam_v_multi_head_attention_1_query_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp<assignvariableop_96_adam_m_multi_head_attention_1_query_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp<assignvariableop_97_adam_v_multi_head_attention_1_query_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp<assignvariableop_98_adam_m_multi_head_attention_1_key_kernelIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp<assignvariableop_99_adam_v_multi_head_attention_1_key_kernelIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp;assignvariableop_100_adam_m_multi_head_attention_1_key_biasIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp;assignvariableop_101_adam_v_multi_head_attention_1_key_biasIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp?assignvariableop_102_adam_m_multi_head_attention_1_value_kernelIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp?assignvariableop_103_adam_v_multi_head_attention_1_value_kernelIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp=assignvariableop_104_adam_m_multi_head_attention_1_value_biasIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp=assignvariableop_105_adam_v_multi_head_attention_1_value_biasIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOpJassignvariableop_106_adam_m_multi_head_attention_1_attention_output_kernelIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpJassignvariableop_107_adam_v_multi_head_attention_1_attention_output_kernelIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOpHassignvariableop_108_adam_m_multi_head_attention_1_attention_output_biasIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOpHassignvariableop_109_adam_v_multi_head_attention_1_attention_output_biasIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp7assignvariableop_110_adam_m_layer_normalization_3_gammaIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp7assignvariableop_111_adam_v_layer_normalization_3_gammaIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp6assignvariableop_112_adam_m_layer_normalization_3_betaIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp6assignvariableop_113_adam_v_layer_normalization_3_betaIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp+assignvariableop_114_adam_m_conv1d_1_kernelIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_v_conv1d_1_kernelIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp)assignvariableop_116_adam_m_conv1d_1_biasIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp)assignvariableop_117_adam_v_conv1d_1_biasIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp7assignvariableop_118_adam_m_layer_normalization_4_gammaIdentity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp7assignvariableop_119_adam_v_layer_normalization_4_gammaIdentity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp6assignvariableop_120_adam_m_layer_normalization_4_betaIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp6assignvariableop_121_adam_v_layer_normalization_4_betaIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp?assignvariableop_122_adam_m_multi_head_attention_2_query_kernelIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp?assignvariableop_123_adam_v_multi_head_attention_2_query_kernelIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp=assignvariableop_124_adam_m_multi_head_attention_2_query_biasIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp=assignvariableop_125_adam_v_multi_head_attention_2_query_biasIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp=assignvariableop_126_adam_m_multi_head_attention_2_key_kernelIdentity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp=assignvariableop_127_adam_v_multi_head_attention_2_key_kernelIdentity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp;assignvariableop_128_adam_m_multi_head_attention_2_key_biasIdentity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp;assignvariableop_129_adam_v_multi_head_attention_2_key_biasIdentity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp?assignvariableop_130_adam_m_multi_head_attention_2_value_kernelIdentity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp?assignvariableop_131_adam_v_multi_head_attention_2_value_kernelIdentity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp=assignvariableop_132_adam_m_multi_head_attention_2_value_biasIdentity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp=assignvariableop_133_adam_v_multi_head_attention_2_value_biasIdentity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOpJassignvariableop_134_adam_m_multi_head_attention_2_attention_output_kernelIdentity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOpJassignvariableop_135_adam_v_multi_head_attention_2_attention_output_kernelIdentity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOpHassignvariableop_136_adam_m_multi_head_attention_2_attention_output_biasIdentity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOpHassignvariableop_137_adam_v_multi_head_attention_2_attention_output_biasIdentity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp7assignvariableop_138_adam_m_layer_normalization_5_gammaIdentity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_v_layer_normalization_5_gammaIdentity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOp6assignvariableop_140_adam_m_layer_normalization_5_betaIdentity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOp6assignvariableop_141_adam_v_layer_normalization_5_betaIdentity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOp+assignvariableop_142_adam_m_conv1d_2_kernelIdentity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_v_conv1d_2_kernelIdentity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_m_conv1d_2_biasIdentity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOp)assignvariableop_145_adam_v_conv1d_2_biasIdentity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOp7assignvariableop_146_adam_m_layer_normalization_6_gammaIdentity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOp7assignvariableop_147_adam_v_layer_normalization_6_gammaIdentity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOp6assignvariableop_148_adam_m_layer_normalization_6_betaIdentity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOp6assignvariableop_149_adam_v_layer_normalization_6_betaIdentity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOp?assignvariableop_150_adam_m_multi_head_attention_3_query_kernelIdentity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOp?assignvariableop_151_adam_v_multi_head_attention_3_query_kernelIdentity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOp=assignvariableop_152_adam_m_multi_head_attention_3_query_biasIdentity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOp=assignvariableop_153_adam_v_multi_head_attention_3_query_biasIdentity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_154AssignVariableOp=assignvariableop_154_adam_m_multi_head_attention_3_key_kernelIdentity_154:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_155AssignVariableOp=assignvariableop_155_adam_v_multi_head_attention_3_key_kernelIdentity_155:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_156AssignVariableOp;assignvariableop_156_adam_m_multi_head_attention_3_key_biasIdentity_156:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_157AssignVariableOp;assignvariableop_157_adam_v_multi_head_attention_3_key_biasIdentity_157:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_158AssignVariableOp?assignvariableop_158_adam_m_multi_head_attention_3_value_kernelIdentity_158:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_159AssignVariableOp?assignvariableop_159_adam_v_multi_head_attention_3_value_kernelIdentity_159:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_160AssignVariableOp=assignvariableop_160_adam_m_multi_head_attention_3_value_biasIdentity_160:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_161AssignVariableOp=assignvariableop_161_adam_v_multi_head_attention_3_value_biasIdentity_161:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_162AssignVariableOpJassignvariableop_162_adam_m_multi_head_attention_3_attention_output_kernelIdentity_162:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_163AssignVariableOpJassignvariableop_163_adam_v_multi_head_attention_3_attention_output_kernelIdentity_163:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_164AssignVariableOpHassignvariableop_164_adam_m_multi_head_attention_3_attention_output_biasIdentity_164:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_165AssignVariableOpHassignvariableop_165_adam_v_multi_head_attention_3_attention_output_biasIdentity_165:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_166AssignVariableOp7assignvariableop_166_adam_m_layer_normalization_7_gammaIdentity_166:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_167AssignVariableOp7assignvariableop_167_adam_v_layer_normalization_7_gammaIdentity_167:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_168AssignVariableOp6assignvariableop_168_adam_m_layer_normalization_7_betaIdentity_168:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_169AssignVariableOp6assignvariableop_169_adam_v_layer_normalization_7_betaIdentity_169:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_170AssignVariableOp+assignvariableop_170_adam_m_conv1d_3_kernelIdentity_170:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_171AssignVariableOp+assignvariableop_171_adam_v_conv1d_3_kernelIdentity_171:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_172AssignVariableOp)assignvariableop_172_adam_m_conv1d_3_biasIdentity_172:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_173AssignVariableOp)assignvariableop_173_adam_v_conv1d_3_biasIdentity_173:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_174AssignVariableOp(assignvariableop_174_adam_m_dense_kernelIdentity_174:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_175AssignVariableOp(assignvariableop_175_adam_v_dense_kernelIdentity_175:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_176AssignVariableOp&assignvariableop_176_adam_m_dense_biasIdentity_176:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_177AssignVariableOp&assignvariableop_177_adam_v_dense_biasIdentity_177:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_178AssignVariableOp*assignvariableop_178_adam_m_dense_1_kernelIdentity_178:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_179AssignVariableOp*assignvariableop_179_adam_v_dense_1_kernelIdentity_179:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_180AssignVariableOp(assignvariableop_180_adam_m_dense_1_biasIdentity_180:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_181IdentityRestoreV2:tensors:181"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_181AssignVariableOp(assignvariableop_181_adam_v_dense_1_biasIdentity_181:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_182IdentityRestoreV2:tensors:182"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_182AssignVariableOpassignvariableop_182_total_1Identity_182:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_183IdentityRestoreV2:tensors:183"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_183AssignVariableOpassignvariableop_183_count_1Identity_183:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_184IdentityRestoreV2:tensors:184"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_184AssignVariableOpassignvariableop_184_totalIdentity_184:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_185IdentityRestoreV2:tensors:185"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_185AssignVariableOpassignvariableop_185_countIdentity_185:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �!
Identity_186Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_187IdentityIdentity_186:output:0^NoOp_1*
T0*
_output_shapes
: � 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_187Identity_187:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722,
AssignVariableOp_173AssignVariableOp_1732,
AssignVariableOp_174AssignVariableOp_1742,
AssignVariableOp_175AssignVariableOp_1752,
AssignVariableOp_176AssignVariableOp_1762,
AssignVariableOp_177AssignVariableOp_1772,
AssignVariableOp_178AssignVariableOp_1782,
AssignVariableOp_179AssignVariableOp_1792*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_180AssignVariableOp_1802,
AssignVariableOp_181AssignVariableOp_1812,
AssignVariableOp_182AssignVariableOp_1822,
AssignVariableOp_183AssignVariableOp_1832,
AssignVariableOp_184AssignVariableOp_1842,
AssignVariableOp_185AssignVariableOp_1852*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:&�!

_user_specified_namecount:&�!

_user_specified_nametotal:(�#
!
_user_specified_name	count_1:(�#
!
_user_specified_name	total_1:4�/
-
_user_specified_nameAdam/v/dense_1/bias:4�/
-
_user_specified_nameAdam/m/dense_1/bias:6�1
/
_user_specified_nameAdam/v/dense_1/kernel:6�1
/
_user_specified_nameAdam/m/dense_1/kernel:2�-
+
_user_specified_nameAdam/v/dense/bias:2�-
+
_user_specified_nameAdam/m/dense/bias:4�/
-
_user_specified_nameAdam/v/dense/kernel:4�/
-
_user_specified_nameAdam/m/dense/kernel:5�0
.
_user_specified_nameAdam/v/conv1d_3/bias:5�0
.
_user_specified_nameAdam/m/conv1d_3/bias:7�2
0
_user_specified_nameAdam/v/conv1d_3/kernel:7�2
0
_user_specified_nameAdam/m/conv1d_3/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_7/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_7/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_7/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_7/gamma:T�O
M
_user_specified_name53Adam/v/multi_head_attention_3/attention_output/bias:T�O
M
_user_specified_name53Adam/m/multi_head_attention_3/attention_output/bias:V�Q
O
_user_specified_name75Adam/v/multi_head_attention_3/attention_output/kernel:V�Q
O
_user_specified_name75Adam/m/multi_head_attention_3/attention_output/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/value/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/value/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_3/value/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_3/value/kernel:G�B
@
_user_specified_name(&Adam/v/multi_head_attention_3/key/bias:G�B
@
_user_specified_name(&Adam/m/multi_head_attention_3/key/bias:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/key/kernel:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/key/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/query/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/query/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_3/query/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_3/query/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_6/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_6/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_6/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_6/gamma:5�0
.
_user_specified_nameAdam/v/conv1d_2/bias:5�0
.
_user_specified_nameAdam/m/conv1d_2/bias:7�2
0
_user_specified_nameAdam/v/conv1d_2/kernel:7�2
0
_user_specified_nameAdam/m/conv1d_2/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_5/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_5/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_5/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_5/gamma:T�O
M
_user_specified_name53Adam/v/multi_head_attention_2/attention_output/bias:T�O
M
_user_specified_name53Adam/m/multi_head_attention_2/attention_output/bias:V�Q
O
_user_specified_name75Adam/v/multi_head_attention_2/attention_output/kernel:V�Q
O
_user_specified_name75Adam/m/multi_head_attention_2/attention_output/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_2/value/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_2/value/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_2/value/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_2/value/kernel:G�B
@
_user_specified_name(&Adam/v/multi_head_attention_2/key/bias:G�B
@
_user_specified_name(&Adam/m/multi_head_attention_2/key/bias:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_2/key/kernel:HD
B
_user_specified_name*(Adam/m/multi_head_attention_2/key/kernel:H~D
B
_user_specified_name*(Adam/v/multi_head_attention_2/query/bias:H}D
B
_user_specified_name*(Adam/m/multi_head_attention_2/query/bias:J|F
D
_user_specified_name,*Adam/v/multi_head_attention_2/query/kernel:J{F
D
_user_specified_name,*Adam/m/multi_head_attention_2/query/kernel:Az=
;
_user_specified_name#!Adam/v/layer_normalization_4/beta:Ay=
;
_user_specified_name#!Adam/m/layer_normalization_4/beta:Bx>
<
_user_specified_name$"Adam/v/layer_normalization_4/gamma:Bw>
<
_user_specified_name$"Adam/m/layer_normalization_4/gamma:4v0
.
_user_specified_nameAdam/v/conv1d_1/bias:4u0
.
_user_specified_nameAdam/m/conv1d_1/bias:6t2
0
_user_specified_nameAdam/v/conv1d_1/kernel:6s2
0
_user_specified_nameAdam/m/conv1d_1/kernel:Ar=
;
_user_specified_name#!Adam/v/layer_normalization_3/beta:Aq=
;
_user_specified_name#!Adam/m/layer_normalization_3/beta:Bp>
<
_user_specified_name$"Adam/v/layer_normalization_3/gamma:Bo>
<
_user_specified_name$"Adam/m/layer_normalization_3/gamma:SnO
M
_user_specified_name53Adam/v/multi_head_attention_1/attention_output/bias:SmO
M
_user_specified_name53Adam/m/multi_head_attention_1/attention_output/bias:UlQ
O
_user_specified_name75Adam/v/multi_head_attention_1/attention_output/kernel:UkQ
O
_user_specified_name75Adam/m/multi_head_attention_1/attention_output/kernel:HjD
B
_user_specified_name*(Adam/v/multi_head_attention_1/value/bias:HiD
B
_user_specified_name*(Adam/m/multi_head_attention_1/value/bias:JhF
D
_user_specified_name,*Adam/v/multi_head_attention_1/value/kernel:JgF
D
_user_specified_name,*Adam/m/multi_head_attention_1/value/kernel:FfB
@
_user_specified_name(&Adam/v/multi_head_attention_1/key/bias:FeB
@
_user_specified_name(&Adam/m/multi_head_attention_1/key/bias:HdD
B
_user_specified_name*(Adam/v/multi_head_attention_1/key/kernel:HcD
B
_user_specified_name*(Adam/m/multi_head_attention_1/key/kernel:HbD
B
_user_specified_name*(Adam/v/multi_head_attention_1/query/bias:HaD
B
_user_specified_name*(Adam/m/multi_head_attention_1/query/bias:J`F
D
_user_specified_name,*Adam/v/multi_head_attention_1/query/kernel:J_F
D
_user_specified_name,*Adam/m/multi_head_attention_1/query/kernel:A^=
;
_user_specified_name#!Adam/v/layer_normalization_2/beta:A]=
;
_user_specified_name#!Adam/m/layer_normalization_2/beta:B\>
<
_user_specified_name$"Adam/v/layer_normalization_2/gamma:B[>
<
_user_specified_name$"Adam/m/layer_normalization_2/gamma:2Z.
,
_user_specified_nameAdam/v/conv1d/bias:2Y.
,
_user_specified_nameAdam/m/conv1d/bias:4X0
.
_user_specified_nameAdam/v/conv1d/kernel:4W0
.
_user_specified_nameAdam/m/conv1d/kernel:AV=
;
_user_specified_name#!Adam/v/layer_normalization_1/beta:AU=
;
_user_specified_name#!Adam/m/layer_normalization_1/beta:BT>
<
_user_specified_name$"Adam/v/layer_normalization_1/gamma:BS>
<
_user_specified_name$"Adam/m/layer_normalization_1/gamma:QRM
K
_user_specified_name31Adam/v/multi_head_attention/attention_output/bias:QQM
K
_user_specified_name31Adam/m/multi_head_attention/attention_output/bias:SPO
M
_user_specified_name53Adam/v/multi_head_attention/attention_output/kernel:SOO
M
_user_specified_name53Adam/m/multi_head_attention/attention_output/kernel:FNB
@
_user_specified_name(&Adam/v/multi_head_attention/value/bias:FMB
@
_user_specified_name(&Adam/m/multi_head_attention/value/bias:HLD
B
_user_specified_name*(Adam/v/multi_head_attention/value/kernel:HKD
B
_user_specified_name*(Adam/m/multi_head_attention/value/kernel:DJ@
>
_user_specified_name&$Adam/v/multi_head_attention/key/bias:DI@
>
_user_specified_name&$Adam/m/multi_head_attention/key/bias:FHB
@
_user_specified_name(&Adam/v/multi_head_attention/key/kernel:FGB
@
_user_specified_name(&Adam/m/multi_head_attention/key/kernel:FFB
@
_user_specified_name(&Adam/v/multi_head_attention/query/bias:FEB
@
_user_specified_name(&Adam/m/multi_head_attention/query/bias:HDD
B
_user_specified_name*(Adam/v/multi_head_attention/query/kernel:HCD
B
_user_specified_name*(Adam/m/multi_head_attention/query/kernel:?B;
9
_user_specified_name!Adam/v/layer_normalization/beta:?A;
9
_user_specified_name!Adam/m/layer_normalization/beta:@@<
:
_user_specified_name" Adam/v/layer_normalization/gamma:@?<
:
_user_specified_name" Adam/m/layer_normalization/gamma:->)
'
_user_specified_namelearning_rate:)=%
#
_user_specified_name	iteration:L<H
F
_user_specified_name.,multi_head_attention_3/attention_output/bias:N;J
H
_user_specified_name0.multi_head_attention_3/attention_output/kernel:A:=
;
_user_specified_name#!multi_head_attention_3/value/bias:C9?
=
_user_specified_name%#multi_head_attention_3/value/kernel:?8;
9
_user_specified_name!multi_head_attention_3/key/bias:A7=
;
_user_specified_name#!multi_head_attention_3/key/kernel:A6=
;
_user_specified_name#!multi_head_attention_3/query/bias:C5?
=
_user_specified_name%#multi_head_attention_3/query/kernel:L4H
F
_user_specified_name.,multi_head_attention_2/attention_output/bias:N3J
H
_user_specified_name0.multi_head_attention_2/attention_output/kernel:A2=
;
_user_specified_name#!multi_head_attention_2/value/bias:C1?
=
_user_specified_name%#multi_head_attention_2/value/kernel:?0;
9
_user_specified_name!multi_head_attention_2/key/bias:A/=
;
_user_specified_name#!multi_head_attention_2/key/kernel:A.=
;
_user_specified_name#!multi_head_attention_2/query/bias:C-?
=
_user_specified_name%#multi_head_attention_2/query/kernel:L,H
F
_user_specified_name.,multi_head_attention_1/attention_output/bias:N+J
H
_user_specified_name0.multi_head_attention_1/attention_output/kernel:A*=
;
_user_specified_name#!multi_head_attention_1/value/bias:C)?
=
_user_specified_name%#multi_head_attention_1/value/kernel:?(;
9
_user_specified_name!multi_head_attention_1/key/bias:A'=
;
_user_specified_name#!multi_head_attention_1/key/kernel:A&=
;
_user_specified_name#!multi_head_attention_1/query/bias:C%?
=
_user_specified_name%#multi_head_attention_1/query/kernel:J$F
D
_user_specified_name,*multi_head_attention/attention_output/bias:L#H
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?";
9
_user_specified_name!multi_head_attention/value/bias:A!=
;
_user_specified_name#!multi_head_attention/value/kernel:= 9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:-)
'
_user_specified_nameconv1d_3/bias:/+
)
_user_specified_nameconv1d_3/kernel::6
4
_user_specified_namelayer_normalization_7/beta:;7
5
_user_specified_namelayer_normalization_7/gamma::6
4
_user_specified_namelayer_normalization_6/beta:;7
5
_user_specified_namelayer_normalization_6/gamma:-)
'
_user_specified_nameconv1d_2/bias:/+
)
_user_specified_nameconv1d_2/kernel::6
4
_user_specified_namelayer_normalization_5/beta:;7
5
_user_specified_namelayer_normalization_5/gamma::6
4
_user_specified_namelayer_normalization_4/beta:;7
5
_user_specified_namelayer_normalization_4/gamma:-)
'
_user_specified_nameconv1d_1/bias:/+
)
_user_specified_nameconv1d_1/kernel::
6
4
_user_specified_namelayer_normalization_3/beta:;	7
5
_user_specified_namelayer_normalization_3/gamma::6
4
_user_specified_namelayer_normalization_2/beta:;7
5
_user_specified_namelayer_normalization_2/gamma:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv1d/kernel::6
4
_user_specified_namelayer_normalization_1/beta:;7
5
_user_specified_namelayer_normalization_1/gamma:84
2
_user_specified_namelayer_normalization/beta:95
3
_user_specified_namelayer_normalization/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_239526

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
˸
��
__inference__traced_save_240969
file_prefix>
0read_disablecopyonread_layer_normalization_gamma:?
1read_1_disablecopyonread_layer_normalization_beta:B
4read_2_disablecopyonread_layer_normalization_1_gamma:A
3read_3_disablecopyonread_layer_normalization_1_beta:<
&read_4_disablecopyonread_conv1d_kernel:2
$read_5_disablecopyonread_conv1d_bias:B
4read_6_disablecopyonread_layer_normalization_2_gamma:A
3read_7_disablecopyonread_layer_normalization_2_beta:B
4read_8_disablecopyonread_layer_normalization_3_gamma:A
3read_9_disablecopyonread_layer_normalization_3_beta:?
)read_10_disablecopyonread_conv1d_1_kernel:5
'read_11_disablecopyonread_conv1d_1_bias:C
5read_12_disablecopyonread_layer_normalization_4_gamma:B
4read_13_disablecopyonread_layer_normalization_4_beta:C
5read_14_disablecopyonread_layer_normalization_5_gamma:B
4read_15_disablecopyonread_layer_normalization_5_beta:?
)read_16_disablecopyonread_conv1d_2_kernel:5
'read_17_disablecopyonread_conv1d_2_bias:C
5read_18_disablecopyonread_layer_normalization_6_gamma:B
4read_19_disablecopyonread_layer_normalization_6_beta:C
5read_20_disablecopyonread_layer_normalization_7_gamma:B
4read_21_disablecopyonread_layer_normalization_7_beta:?
)read_22_disablecopyonread_conv1d_3_kernel:5
'read_23_disablecopyonread_conv1d_3_bias:9
&read_24_disablecopyonread_dense_kernel:	�3
$read_25_disablecopyonread_dense_bias:	�;
(read_26_disablecopyonread_dense_1_kernel:	�4
&read_27_disablecopyonread_dense_1_bias:Q
;read_28_disablecopyonread_multi_head_attention_query_kernel:@K
9read_29_disablecopyonread_multi_head_attention_query_bias:@O
9read_30_disablecopyonread_multi_head_attention_key_kernel:@I
7read_31_disablecopyonread_multi_head_attention_key_bias:@Q
;read_32_disablecopyonread_multi_head_attention_value_kernel:@K
9read_33_disablecopyonread_multi_head_attention_value_bias:@\
Fread_34_disablecopyonread_multi_head_attention_attention_output_kernel:@R
Dread_35_disablecopyonread_multi_head_attention_attention_output_bias:S
=read_36_disablecopyonread_multi_head_attention_1_query_kernel:@M
;read_37_disablecopyonread_multi_head_attention_1_query_bias:@Q
;read_38_disablecopyonread_multi_head_attention_1_key_kernel:@K
9read_39_disablecopyonread_multi_head_attention_1_key_bias:@S
=read_40_disablecopyonread_multi_head_attention_1_value_kernel:@M
;read_41_disablecopyonread_multi_head_attention_1_value_bias:@^
Hread_42_disablecopyonread_multi_head_attention_1_attention_output_kernel:@T
Fread_43_disablecopyonread_multi_head_attention_1_attention_output_bias:S
=read_44_disablecopyonread_multi_head_attention_2_query_kernel:@M
;read_45_disablecopyonread_multi_head_attention_2_query_bias:@Q
;read_46_disablecopyonread_multi_head_attention_2_key_kernel:@K
9read_47_disablecopyonread_multi_head_attention_2_key_bias:@S
=read_48_disablecopyonread_multi_head_attention_2_value_kernel:@M
;read_49_disablecopyonread_multi_head_attention_2_value_bias:@^
Hread_50_disablecopyonread_multi_head_attention_2_attention_output_kernel:@T
Fread_51_disablecopyonread_multi_head_attention_2_attention_output_bias:S
=read_52_disablecopyonread_multi_head_attention_3_query_kernel:@M
;read_53_disablecopyonread_multi_head_attention_3_query_bias:@Q
;read_54_disablecopyonread_multi_head_attention_3_key_kernel:@K
9read_55_disablecopyonread_multi_head_attention_3_key_bias:@S
=read_56_disablecopyonread_multi_head_attention_3_value_kernel:@M
;read_57_disablecopyonread_multi_head_attention_3_value_bias:@^
Hread_58_disablecopyonread_multi_head_attention_3_attention_output_kernel:@T
Fread_59_disablecopyonread_multi_head_attention_3_attention_output_bias:-
#read_60_disablecopyonread_iteration:	 1
'read_61_disablecopyonread_learning_rate: H
:read_62_disablecopyonread_adam_m_layer_normalization_gamma:H
:read_63_disablecopyonread_adam_v_layer_normalization_gamma:G
9read_64_disablecopyonread_adam_m_layer_normalization_beta:G
9read_65_disablecopyonread_adam_v_layer_normalization_beta:X
Bread_66_disablecopyonread_adam_m_multi_head_attention_query_kernel:@X
Bread_67_disablecopyonread_adam_v_multi_head_attention_query_kernel:@R
@read_68_disablecopyonread_adam_m_multi_head_attention_query_bias:@R
@read_69_disablecopyonread_adam_v_multi_head_attention_query_bias:@V
@read_70_disablecopyonread_adam_m_multi_head_attention_key_kernel:@V
@read_71_disablecopyonread_adam_v_multi_head_attention_key_kernel:@P
>read_72_disablecopyonread_adam_m_multi_head_attention_key_bias:@P
>read_73_disablecopyonread_adam_v_multi_head_attention_key_bias:@X
Bread_74_disablecopyonread_adam_m_multi_head_attention_value_kernel:@X
Bread_75_disablecopyonread_adam_v_multi_head_attention_value_kernel:@R
@read_76_disablecopyonread_adam_m_multi_head_attention_value_bias:@R
@read_77_disablecopyonread_adam_v_multi_head_attention_value_bias:@c
Mread_78_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel:@c
Mread_79_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel:@Y
Kread_80_disablecopyonread_adam_m_multi_head_attention_attention_output_bias:Y
Kread_81_disablecopyonread_adam_v_multi_head_attention_attention_output_bias:J
<read_82_disablecopyonread_adam_m_layer_normalization_1_gamma:J
<read_83_disablecopyonread_adam_v_layer_normalization_1_gamma:I
;read_84_disablecopyonread_adam_m_layer_normalization_1_beta:I
;read_85_disablecopyonread_adam_v_layer_normalization_1_beta:D
.read_86_disablecopyonread_adam_m_conv1d_kernel:D
.read_87_disablecopyonread_adam_v_conv1d_kernel::
,read_88_disablecopyonread_adam_m_conv1d_bias::
,read_89_disablecopyonread_adam_v_conv1d_bias:J
<read_90_disablecopyonread_adam_m_layer_normalization_2_gamma:J
<read_91_disablecopyonread_adam_v_layer_normalization_2_gamma:I
;read_92_disablecopyonread_adam_m_layer_normalization_2_beta:I
;read_93_disablecopyonread_adam_v_layer_normalization_2_beta:Z
Dread_94_disablecopyonread_adam_m_multi_head_attention_1_query_kernel:@Z
Dread_95_disablecopyonread_adam_v_multi_head_attention_1_query_kernel:@T
Bread_96_disablecopyonread_adam_m_multi_head_attention_1_query_bias:@T
Bread_97_disablecopyonread_adam_v_multi_head_attention_1_query_bias:@X
Bread_98_disablecopyonread_adam_m_multi_head_attention_1_key_kernel:@X
Bread_99_disablecopyonread_adam_v_multi_head_attention_1_key_kernel:@S
Aread_100_disablecopyonread_adam_m_multi_head_attention_1_key_bias:@S
Aread_101_disablecopyonread_adam_v_multi_head_attention_1_key_bias:@[
Eread_102_disablecopyonread_adam_m_multi_head_attention_1_value_kernel:@[
Eread_103_disablecopyonread_adam_v_multi_head_attention_1_value_kernel:@U
Cread_104_disablecopyonread_adam_m_multi_head_attention_1_value_bias:@U
Cread_105_disablecopyonread_adam_v_multi_head_attention_1_value_bias:@f
Pread_106_disablecopyonread_adam_m_multi_head_attention_1_attention_output_kernel:@f
Pread_107_disablecopyonread_adam_v_multi_head_attention_1_attention_output_kernel:@\
Nread_108_disablecopyonread_adam_m_multi_head_attention_1_attention_output_bias:\
Nread_109_disablecopyonread_adam_v_multi_head_attention_1_attention_output_bias:K
=read_110_disablecopyonread_adam_m_layer_normalization_3_gamma:K
=read_111_disablecopyonread_adam_v_layer_normalization_3_gamma:J
<read_112_disablecopyonread_adam_m_layer_normalization_3_beta:J
<read_113_disablecopyonread_adam_v_layer_normalization_3_beta:G
1read_114_disablecopyonread_adam_m_conv1d_1_kernel:G
1read_115_disablecopyonread_adam_v_conv1d_1_kernel:=
/read_116_disablecopyonread_adam_m_conv1d_1_bias:=
/read_117_disablecopyonread_adam_v_conv1d_1_bias:K
=read_118_disablecopyonread_adam_m_layer_normalization_4_gamma:K
=read_119_disablecopyonread_adam_v_layer_normalization_4_gamma:J
<read_120_disablecopyonread_adam_m_layer_normalization_4_beta:J
<read_121_disablecopyonread_adam_v_layer_normalization_4_beta:[
Eread_122_disablecopyonread_adam_m_multi_head_attention_2_query_kernel:@[
Eread_123_disablecopyonread_adam_v_multi_head_attention_2_query_kernel:@U
Cread_124_disablecopyonread_adam_m_multi_head_attention_2_query_bias:@U
Cread_125_disablecopyonread_adam_v_multi_head_attention_2_query_bias:@Y
Cread_126_disablecopyonread_adam_m_multi_head_attention_2_key_kernel:@Y
Cread_127_disablecopyonread_adam_v_multi_head_attention_2_key_kernel:@S
Aread_128_disablecopyonread_adam_m_multi_head_attention_2_key_bias:@S
Aread_129_disablecopyonread_adam_v_multi_head_attention_2_key_bias:@[
Eread_130_disablecopyonread_adam_m_multi_head_attention_2_value_kernel:@[
Eread_131_disablecopyonread_adam_v_multi_head_attention_2_value_kernel:@U
Cread_132_disablecopyonread_adam_m_multi_head_attention_2_value_bias:@U
Cread_133_disablecopyonread_adam_v_multi_head_attention_2_value_bias:@f
Pread_134_disablecopyonread_adam_m_multi_head_attention_2_attention_output_kernel:@f
Pread_135_disablecopyonread_adam_v_multi_head_attention_2_attention_output_kernel:@\
Nread_136_disablecopyonread_adam_m_multi_head_attention_2_attention_output_bias:\
Nread_137_disablecopyonread_adam_v_multi_head_attention_2_attention_output_bias:K
=read_138_disablecopyonread_adam_m_layer_normalization_5_gamma:K
=read_139_disablecopyonread_adam_v_layer_normalization_5_gamma:J
<read_140_disablecopyonread_adam_m_layer_normalization_5_beta:J
<read_141_disablecopyonread_adam_v_layer_normalization_5_beta:G
1read_142_disablecopyonread_adam_m_conv1d_2_kernel:G
1read_143_disablecopyonread_adam_v_conv1d_2_kernel:=
/read_144_disablecopyonread_adam_m_conv1d_2_bias:=
/read_145_disablecopyonread_adam_v_conv1d_2_bias:K
=read_146_disablecopyonread_adam_m_layer_normalization_6_gamma:K
=read_147_disablecopyonread_adam_v_layer_normalization_6_gamma:J
<read_148_disablecopyonread_adam_m_layer_normalization_6_beta:J
<read_149_disablecopyonread_adam_v_layer_normalization_6_beta:[
Eread_150_disablecopyonread_adam_m_multi_head_attention_3_query_kernel:@[
Eread_151_disablecopyonread_adam_v_multi_head_attention_3_query_kernel:@U
Cread_152_disablecopyonread_adam_m_multi_head_attention_3_query_bias:@U
Cread_153_disablecopyonread_adam_v_multi_head_attention_3_query_bias:@Y
Cread_154_disablecopyonread_adam_m_multi_head_attention_3_key_kernel:@Y
Cread_155_disablecopyonread_adam_v_multi_head_attention_3_key_kernel:@S
Aread_156_disablecopyonread_adam_m_multi_head_attention_3_key_bias:@S
Aread_157_disablecopyonread_adam_v_multi_head_attention_3_key_bias:@[
Eread_158_disablecopyonread_adam_m_multi_head_attention_3_value_kernel:@[
Eread_159_disablecopyonread_adam_v_multi_head_attention_3_value_kernel:@U
Cread_160_disablecopyonread_adam_m_multi_head_attention_3_value_bias:@U
Cread_161_disablecopyonread_adam_v_multi_head_attention_3_value_bias:@f
Pread_162_disablecopyonread_adam_m_multi_head_attention_3_attention_output_kernel:@f
Pread_163_disablecopyonread_adam_v_multi_head_attention_3_attention_output_kernel:@\
Nread_164_disablecopyonread_adam_m_multi_head_attention_3_attention_output_bias:\
Nread_165_disablecopyonread_adam_v_multi_head_attention_3_attention_output_bias:K
=read_166_disablecopyonread_adam_m_layer_normalization_7_gamma:K
=read_167_disablecopyonread_adam_v_layer_normalization_7_gamma:J
<read_168_disablecopyonread_adam_m_layer_normalization_7_beta:J
<read_169_disablecopyonread_adam_v_layer_normalization_7_beta:G
1read_170_disablecopyonread_adam_m_conv1d_3_kernel:G
1read_171_disablecopyonread_adam_v_conv1d_3_kernel:=
/read_172_disablecopyonread_adam_m_conv1d_3_bias:=
/read_173_disablecopyonread_adam_v_conv1d_3_bias:A
.read_174_disablecopyonread_adam_m_dense_kernel:	�A
.read_175_disablecopyonread_adam_v_dense_kernel:	�;
,read_176_disablecopyonread_adam_m_dense_bias:	�;
,read_177_disablecopyonread_adam_v_dense_bias:	�C
0read_178_disablecopyonread_adam_m_dense_1_kernel:	�C
0read_179_disablecopyonread_adam_v_dense_1_kernel:	�<
.read_180_disablecopyonread_adam_m_dense_1_bias:<
.read_181_disablecopyonread_adam_v_dense_1_bias:,
"read_182_disablecopyonread_total_1: ,
"read_183_disablecopyonread_count_1: *
 read_184_disablecopyonread_total: *
 read_185_disablecopyonread_count: 
savev2_const
identity_373��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_118/DisableCopyOnRead�Read_118/ReadVariableOp�Read_119/DisableCopyOnRead�Read_119/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_120/DisableCopyOnRead�Read_120/ReadVariableOp�Read_121/DisableCopyOnRead�Read_121/ReadVariableOp�Read_122/DisableCopyOnRead�Read_122/ReadVariableOp�Read_123/DisableCopyOnRead�Read_123/ReadVariableOp�Read_124/DisableCopyOnRead�Read_124/ReadVariableOp�Read_125/DisableCopyOnRead�Read_125/ReadVariableOp�Read_126/DisableCopyOnRead�Read_126/ReadVariableOp�Read_127/DisableCopyOnRead�Read_127/ReadVariableOp�Read_128/DisableCopyOnRead�Read_128/ReadVariableOp�Read_129/DisableCopyOnRead�Read_129/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_130/DisableCopyOnRead�Read_130/ReadVariableOp�Read_131/DisableCopyOnRead�Read_131/ReadVariableOp�Read_132/DisableCopyOnRead�Read_132/ReadVariableOp�Read_133/DisableCopyOnRead�Read_133/ReadVariableOp�Read_134/DisableCopyOnRead�Read_134/ReadVariableOp�Read_135/DisableCopyOnRead�Read_135/ReadVariableOp�Read_136/DisableCopyOnRead�Read_136/ReadVariableOp�Read_137/DisableCopyOnRead�Read_137/ReadVariableOp�Read_138/DisableCopyOnRead�Read_138/ReadVariableOp�Read_139/DisableCopyOnRead�Read_139/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_140/DisableCopyOnRead�Read_140/ReadVariableOp�Read_141/DisableCopyOnRead�Read_141/ReadVariableOp�Read_142/DisableCopyOnRead�Read_142/ReadVariableOp�Read_143/DisableCopyOnRead�Read_143/ReadVariableOp�Read_144/DisableCopyOnRead�Read_144/ReadVariableOp�Read_145/DisableCopyOnRead�Read_145/ReadVariableOp�Read_146/DisableCopyOnRead�Read_146/ReadVariableOp�Read_147/DisableCopyOnRead�Read_147/ReadVariableOp�Read_148/DisableCopyOnRead�Read_148/ReadVariableOp�Read_149/DisableCopyOnRead�Read_149/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_150/DisableCopyOnRead�Read_150/ReadVariableOp�Read_151/DisableCopyOnRead�Read_151/ReadVariableOp�Read_152/DisableCopyOnRead�Read_152/ReadVariableOp�Read_153/DisableCopyOnRead�Read_153/ReadVariableOp�Read_154/DisableCopyOnRead�Read_154/ReadVariableOp�Read_155/DisableCopyOnRead�Read_155/ReadVariableOp�Read_156/DisableCopyOnRead�Read_156/ReadVariableOp�Read_157/DisableCopyOnRead�Read_157/ReadVariableOp�Read_158/DisableCopyOnRead�Read_158/ReadVariableOp�Read_159/DisableCopyOnRead�Read_159/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_160/DisableCopyOnRead�Read_160/ReadVariableOp�Read_161/DisableCopyOnRead�Read_161/ReadVariableOp�Read_162/DisableCopyOnRead�Read_162/ReadVariableOp�Read_163/DisableCopyOnRead�Read_163/ReadVariableOp�Read_164/DisableCopyOnRead�Read_164/ReadVariableOp�Read_165/DisableCopyOnRead�Read_165/ReadVariableOp�Read_166/DisableCopyOnRead�Read_166/ReadVariableOp�Read_167/DisableCopyOnRead�Read_167/ReadVariableOp�Read_168/DisableCopyOnRead�Read_168/ReadVariableOp�Read_169/DisableCopyOnRead�Read_169/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_170/DisableCopyOnRead�Read_170/ReadVariableOp�Read_171/DisableCopyOnRead�Read_171/ReadVariableOp�Read_172/DisableCopyOnRead�Read_172/ReadVariableOp�Read_173/DisableCopyOnRead�Read_173/ReadVariableOp�Read_174/DisableCopyOnRead�Read_174/ReadVariableOp�Read_175/DisableCopyOnRead�Read_175/ReadVariableOp�Read_176/DisableCopyOnRead�Read_176/ReadVariableOp�Read_177/DisableCopyOnRead�Read_177/ReadVariableOp�Read_178/DisableCopyOnRead�Read_178/ReadVariableOp�Read_179/DisableCopyOnRead�Read_179/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_180/DisableCopyOnRead�Read_180/ReadVariableOp�Read_181/DisableCopyOnRead�Read_181/ReadVariableOp�Read_182/DisableCopyOnRead�Read_182/ReadVariableOp�Read_183/DisableCopyOnRead�Read_183/ReadVariableOp�Read_184/DisableCopyOnRead�Read_184/ReadVariableOp�Read_185/DisableCopyOnRead�Read_185/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead0read_disablecopyonread_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp0read_disablecopyonread_layer_normalization_gamma^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead1read_1_disablecopyonread_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp1read_1_disablecopyonread_layer_normalization_beta^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_layer_normalization_1_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead3read_3_disablecopyonread_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp3read_3_disablecopyonread_layer_normalization_1_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_conv1d_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0q

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:g

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*"
_output_shapes
:x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_conv1d_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead4read_6_disablecopyonread_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp4read_6_disablecopyonread_layer_normalization_2_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_7/DisableCopyOnReadDisableCopyOnRead3read_7_disablecopyonread_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp3read_7_disablecopyonread_layer_normalization_2_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_layer_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_layer_normalization_3_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_layer_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_layer_normalization_3_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_conv1d_1_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*"
_output_shapes
:|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_conv1d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_conv1d_1_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead5read_12_disablecopyonread_layer_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp5read_12_disablecopyonread_layer_normalization_4_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead4read_13_disablecopyonread_layer_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp4read_13_disablecopyonread_layer_normalization_4_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_layer_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_layer_normalization_5_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_layer_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_layer_normalization_5_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_conv1d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_conv1d_2_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*"
_output_shapes
:|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_conv1d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_conv1d_2_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead5read_18_disablecopyonread_layer_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp5read_18_disablecopyonread_layer_normalization_6_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_layer_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_layer_normalization_6_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_layer_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_layer_normalization_7_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_layer_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_layer_normalization_7_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_conv1d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_conv1d_3_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*"
_output_shapes
:|
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_conv1d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_conv1d_3_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_24/DisableCopyOnReadDisableCopyOnRead&read_24_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp&read_24_disablecopyonread_dense_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_dense_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_26/DisableCopyOnReadDisableCopyOnRead(read_26_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp(read_26_disablecopyonread_dense_1_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_27/DisableCopyOnReadDisableCopyOnRead&read_27_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp&read_27_disablecopyonread_dense_1_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_multi_head_attention_query_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead9read_29_disablecopyonread_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp9read_29_disablecopyonread_multi_head_attention_query_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_30/DisableCopyOnReadDisableCopyOnRead9read_30_disablecopyonread_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp9read_30_disablecopyonread_multi_head_attention_key_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_31/DisableCopyOnReadDisableCopyOnRead7read_31_disablecopyonread_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp7read_31_disablecopyonread_multi_head_attention_key_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_32/DisableCopyOnReadDisableCopyOnRead;read_32_disablecopyonread_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp;read_32_disablecopyonread_multi_head_attention_value_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_33/DisableCopyOnReadDisableCopyOnRead9read_33_disablecopyonread_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp9read_33_disablecopyonread_multi_head_attention_value_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_34/DisableCopyOnReadDisableCopyOnReadFread_34_disablecopyonread_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpFread_34_disablecopyonread_multi_head_attention_attention_output_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnReadDread_35_disablecopyonread_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpDread_35_disablecopyonread_multi_head_attention_attention_output_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_36/DisableCopyOnReadDisableCopyOnRead=read_36_disablecopyonread_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp=read_36_disablecopyonread_multi_head_attention_1_query_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead;read_37_disablecopyonread_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp;read_37_disablecopyonread_multi_head_attention_1_query_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_38/DisableCopyOnReadDisableCopyOnRead;read_38_disablecopyonread_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp;read_38_disablecopyonread_multi_head_attention_1_key_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead9read_39_disablecopyonread_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp9read_39_disablecopyonread_multi_head_attention_1_key_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_40/DisableCopyOnReadDisableCopyOnRead=read_40_disablecopyonread_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp=read_40_disablecopyonread_multi_head_attention_1_value_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_41/DisableCopyOnReadDisableCopyOnRead;read_41_disablecopyonread_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp;read_41_disablecopyonread_multi_head_attention_1_value_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_42/DisableCopyOnReadDisableCopyOnReadHread_42_disablecopyonread_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpHread_42_disablecopyonread_multi_head_attention_1_attention_output_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_43/DisableCopyOnReadDisableCopyOnReadFread_43_disablecopyonread_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpFread_43_disablecopyonread_multi_head_attention_1_attention_output_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead=read_44_disablecopyonread_multi_head_attention_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp=read_44_disablecopyonread_multi_head_attention_2_query_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_45/DisableCopyOnReadDisableCopyOnRead;read_45_disablecopyonread_multi_head_attention_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp;read_45_disablecopyonread_multi_head_attention_2_query_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_46/DisableCopyOnReadDisableCopyOnRead;read_46_disablecopyonread_multi_head_attention_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp;read_46_disablecopyonread_multi_head_attention_2_key_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_47/DisableCopyOnReadDisableCopyOnRead9read_47_disablecopyonread_multi_head_attention_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp9read_47_disablecopyonread_multi_head_attention_2_key_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_48/DisableCopyOnReadDisableCopyOnRead=read_48_disablecopyonread_multi_head_attention_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp=read_48_disablecopyonread_multi_head_attention_2_value_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0s
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@i
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_49/DisableCopyOnReadDisableCopyOnRead;read_49_disablecopyonread_multi_head_attention_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp;read_49_disablecopyonread_multi_head_attention_2_value_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_50/DisableCopyOnReadDisableCopyOnReadHread_50_disablecopyonread_multi_head_attention_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpHread_50_disablecopyonread_multi_head_attention_2_attention_output_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_51/DisableCopyOnReadDisableCopyOnReadFread_51_disablecopyonread_multi_head_attention_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpFread_51_disablecopyonread_multi_head_attention_2_attention_output_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead=read_52_disablecopyonread_multi_head_attention_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp=read_52_disablecopyonread_multi_head_attention_3_query_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_53/DisableCopyOnReadDisableCopyOnRead;read_53_disablecopyonread_multi_head_attention_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp;read_53_disablecopyonread_multi_head_attention_3_query_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_54/DisableCopyOnReadDisableCopyOnRead;read_54_disablecopyonread_multi_head_attention_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp;read_54_disablecopyonread_multi_head_attention_3_key_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_55/DisableCopyOnReadDisableCopyOnRead9read_55_disablecopyonread_multi_head_attention_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp9read_55_disablecopyonread_multi_head_attention_3_key_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_56/DisableCopyOnReadDisableCopyOnRead=read_56_disablecopyonread_multi_head_attention_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp=read_56_disablecopyonread_multi_head_attention_3_value_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_57/DisableCopyOnReadDisableCopyOnRead;read_57_disablecopyonread_multi_head_attention_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp;read_57_disablecopyonread_multi_head_attention_3_value_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_58/DisableCopyOnReadDisableCopyOnReadHread_58_disablecopyonread_multi_head_attention_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpHread_58_disablecopyonread_multi_head_attention_3_attention_output_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_59/DisableCopyOnReadDisableCopyOnReadFread_59_disablecopyonread_multi_head_attention_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpFread_59_disablecopyonread_multi_head_attention_3_attention_output_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_60/DisableCopyOnReadDisableCopyOnRead#read_60_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp#read_60_disablecopyonread_iteration^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_61/DisableCopyOnReadDisableCopyOnRead'read_61_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp'read_61_disablecopyonread_learning_rate^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_62/DisableCopyOnReadDisableCopyOnRead:read_62_disablecopyonread_adam_m_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp:read_62_disablecopyonread_adam_m_layer_normalization_gamma^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_63/DisableCopyOnReadDisableCopyOnRead:read_63_disablecopyonread_adam_v_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp:read_63_disablecopyonread_adam_v_layer_normalization_gamma^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_64/DisableCopyOnReadDisableCopyOnRead9read_64_disablecopyonread_adam_m_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp9read_64_disablecopyonread_adam_m_layer_normalization_beta^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead9read_65_disablecopyonread_adam_v_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp9read_65_disablecopyonread_adam_v_layer_normalization_beta^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnReadBread_66_disablecopyonread_adam_m_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOpBread_66_disablecopyonread_adam_m_multi_head_attention_query_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_67/DisableCopyOnReadDisableCopyOnReadBread_67_disablecopyonread_adam_v_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOpBread_67_disablecopyonread_adam_v_multi_head_attention_query_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_68/DisableCopyOnReadDisableCopyOnRead@read_68_disablecopyonread_adam_m_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp@read_68_disablecopyonread_adam_m_multi_head_attention_query_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_69/DisableCopyOnReadDisableCopyOnRead@read_69_disablecopyonread_adam_v_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp@read_69_disablecopyonread_adam_v_multi_head_attention_query_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_70/DisableCopyOnReadDisableCopyOnRead@read_70_disablecopyonread_adam_m_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp@read_70_disablecopyonread_adam_m_multi_head_attention_key_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_71/DisableCopyOnReadDisableCopyOnRead@read_71_disablecopyonread_adam_v_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp@read_71_disablecopyonread_adam_v_multi_head_attention_key_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_72/DisableCopyOnReadDisableCopyOnRead>read_72_disablecopyonread_adam_m_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp>read_72_disablecopyonread_adam_m_multi_head_attention_key_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_73/DisableCopyOnReadDisableCopyOnRead>read_73_disablecopyonread_adam_v_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp>read_73_disablecopyonread_adam_v_multi_head_attention_key_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_74/DisableCopyOnReadDisableCopyOnReadBread_74_disablecopyonread_adam_m_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOpBread_74_disablecopyonread_adam_m_multi_head_attention_value_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_75/DisableCopyOnReadDisableCopyOnReadBread_75_disablecopyonread_adam_v_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOpBread_75_disablecopyonread_adam_v_multi_head_attention_value_kernel^Read_75/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_76/DisableCopyOnReadDisableCopyOnRead@read_76_disablecopyonread_adam_m_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp@read_76_disablecopyonread_adam_m_multi_head_attention_value_bias^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_77/DisableCopyOnReadDisableCopyOnRead@read_77_disablecopyonread_adam_v_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp@read_77_disablecopyonread_adam_v_multi_head_attention_value_bias^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_78/DisableCopyOnReadDisableCopyOnReadMread_78_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOpMread_78_disablecopyonread_adam_m_multi_head_attention_attention_output_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_79/DisableCopyOnReadDisableCopyOnReadMread_79_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOpMread_79_disablecopyonread_adam_v_multi_head_attention_attention_output_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_80/DisableCopyOnReadDisableCopyOnReadKread_80_disablecopyonread_adam_m_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOpKread_80_disablecopyonread_adam_m_multi_head_attention_attention_output_bias^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_81/DisableCopyOnReadDisableCopyOnReadKread_81_disablecopyonread_adam_v_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOpKread_81_disablecopyonread_adam_v_multi_head_attention_attention_output_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_82/DisableCopyOnReadDisableCopyOnRead<read_82_disablecopyonread_adam_m_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp<read_82_disablecopyonread_adam_m_layer_normalization_1_gamma^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_83/DisableCopyOnReadDisableCopyOnRead<read_83_disablecopyonread_adam_v_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp<read_83_disablecopyonread_adam_v_layer_normalization_1_gamma^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_84/DisableCopyOnReadDisableCopyOnRead;read_84_disablecopyonread_adam_m_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp;read_84_disablecopyonread_adam_m_layer_normalization_1_beta^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_85/DisableCopyOnReadDisableCopyOnRead;read_85_disablecopyonread_adam_v_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp;read_85_disablecopyonread_adam_v_layer_normalization_1_beta^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_86/DisableCopyOnReadDisableCopyOnRead.read_86_disablecopyonread_adam_m_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp.read_86_disablecopyonread_adam_m_conv1d_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_87/DisableCopyOnReadDisableCopyOnRead.read_87_disablecopyonread_adam_v_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp.read_87_disablecopyonread_adam_v_conv1d_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_88/DisableCopyOnReadDisableCopyOnRead,read_88_disablecopyonread_adam_m_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp,read_88_disablecopyonread_adam_m_conv1d_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_89/DisableCopyOnReadDisableCopyOnRead,read_89_disablecopyonread_adam_v_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp,read_89_disablecopyonread_adam_v_conv1d_bias^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_90/DisableCopyOnReadDisableCopyOnRead<read_90_disablecopyonread_adam_m_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp<read_90_disablecopyonread_adam_m_layer_normalization_2_gamma^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_91/DisableCopyOnReadDisableCopyOnRead<read_91_disablecopyonread_adam_v_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp<read_91_disablecopyonread_adam_v_layer_normalization_2_gamma^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_92/DisableCopyOnReadDisableCopyOnRead;read_92_disablecopyonread_adam_m_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp;read_92_disablecopyonread_adam_m_layer_normalization_2_beta^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_93/DisableCopyOnReadDisableCopyOnRead;read_93_disablecopyonread_adam_v_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp;read_93_disablecopyonread_adam_v_layer_normalization_2_beta^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_94/DisableCopyOnReadDisableCopyOnReadDread_94_disablecopyonread_adam_m_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOpDread_94_disablecopyonread_adam_m_multi_head_attention_1_query_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_95/DisableCopyOnReadDisableCopyOnReadDread_95_disablecopyonread_adam_v_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOpDread_95_disablecopyonread_adam_v_multi_head_attention_1_query_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_96/DisableCopyOnReadDisableCopyOnReadBread_96_disablecopyonread_adam_m_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOpBread_96_disablecopyonread_adam_m_multi_head_attention_1_query_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_97/DisableCopyOnReadDisableCopyOnReadBread_97_disablecopyonread_adam_v_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOpBread_97_disablecopyonread_adam_v_multi_head_attention_1_query_bias^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_98/DisableCopyOnReadDisableCopyOnReadBread_98_disablecopyonread_adam_m_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOpBread_98_disablecopyonread_adam_m_multi_head_attention_1_key_kernel^Read_98/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_99/DisableCopyOnReadDisableCopyOnReadBread_99_disablecopyonread_adam_v_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOpBread_99_disablecopyonread_adam_v_multi_head_attention_1_key_kernel^Read_99/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0t
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_100/DisableCopyOnReadDisableCopyOnReadAread_100_disablecopyonread_adam_m_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOpAread_100_disablecopyonread_adam_m_multi_head_attention_1_key_bias^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_101/DisableCopyOnReadDisableCopyOnReadAread_101_disablecopyonread_adam_v_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOpAread_101_disablecopyonread_adam_v_multi_head_attention_1_key_bias^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_102/DisableCopyOnReadDisableCopyOnReadEread_102_disablecopyonread_adam_m_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOpEread_102_disablecopyonread_adam_m_multi_head_attention_1_value_kernel^Read_102/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_103/DisableCopyOnReadDisableCopyOnReadEread_103_disablecopyonread_adam_v_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOpEread_103_disablecopyonread_adam_v_multi_head_attention_1_value_kernel^Read_103/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_104/DisableCopyOnReadDisableCopyOnReadCread_104_disablecopyonread_adam_m_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOpCread_104_disablecopyonread_adam_m_multi_head_attention_1_value_bias^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_105/DisableCopyOnReadDisableCopyOnReadCread_105_disablecopyonread_adam_v_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOpCread_105_disablecopyonread_adam_v_multi_head_attention_1_value_bias^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_106/DisableCopyOnReadDisableCopyOnReadPread_106_disablecopyonread_adam_m_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOpPread_106_disablecopyonread_adam_m_multi_head_attention_1_attention_output_kernel^Read_106/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_107/DisableCopyOnReadDisableCopyOnReadPread_107_disablecopyonread_adam_v_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOpPread_107_disablecopyonread_adam_v_multi_head_attention_1_attention_output_kernel^Read_107/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_108/DisableCopyOnReadDisableCopyOnReadNread_108_disablecopyonread_adam_m_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOpNread_108_disablecopyonread_adam_m_multi_head_attention_1_attention_output_bias^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_109/DisableCopyOnReadDisableCopyOnReadNread_109_disablecopyonread_adam_v_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOpNread_109_disablecopyonread_adam_v_multi_head_attention_1_attention_output_bias^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_110/DisableCopyOnReadDisableCopyOnRead=read_110_disablecopyonread_adam_m_layer_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp=read_110_disablecopyonread_adam_m_layer_normalization_3_gamma^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_111/DisableCopyOnReadDisableCopyOnRead=read_111_disablecopyonread_adam_v_layer_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp=read_111_disablecopyonread_adam_v_layer_normalization_3_gamma^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_112/DisableCopyOnReadDisableCopyOnRead<read_112_disablecopyonread_adam_m_layer_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp<read_112_disablecopyonread_adam_m_layer_normalization_3_beta^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_113/DisableCopyOnReadDisableCopyOnRead<read_113_disablecopyonread_adam_v_layer_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp<read_113_disablecopyonread_adam_v_layer_normalization_3_beta^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_114/DisableCopyOnReadDisableCopyOnRead1read_114_disablecopyonread_adam_m_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOp1read_114_disablecopyonread_adam_m_conv1d_1_kernel^Read_114/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_115/DisableCopyOnReadDisableCopyOnRead1read_115_disablecopyonread_adam_v_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp1read_115_disablecopyonread_adam_v_conv1d_1_kernel^Read_115/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_116/DisableCopyOnReadDisableCopyOnRead/read_116_disablecopyonread_adam_m_conv1d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp/read_116_disablecopyonread_adam_m_conv1d_1_bias^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_117/DisableCopyOnReadDisableCopyOnRead/read_117_disablecopyonread_adam_v_conv1d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp/read_117_disablecopyonread_adam_v_conv1d_1_bias^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_118/DisableCopyOnReadDisableCopyOnRead=read_118_disablecopyonread_adam_m_layer_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_118/ReadVariableOpReadVariableOp=read_118_disablecopyonread_adam_m_layer_normalization_4_gamma^Read_118/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_119/DisableCopyOnReadDisableCopyOnRead=read_119_disablecopyonread_adam_v_layer_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_119/ReadVariableOpReadVariableOp=read_119_disablecopyonread_adam_v_layer_normalization_4_gamma^Read_119/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_120/DisableCopyOnReadDisableCopyOnRead<read_120_disablecopyonread_adam_m_layer_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_120/ReadVariableOpReadVariableOp<read_120_disablecopyonread_adam_m_layer_normalization_4_beta^Read_120/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_121/DisableCopyOnReadDisableCopyOnRead<read_121_disablecopyonread_adam_v_layer_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_121/ReadVariableOpReadVariableOp<read_121_disablecopyonread_adam_v_layer_normalization_4_beta^Read_121/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_122/DisableCopyOnReadDisableCopyOnReadEread_122_disablecopyonread_adam_m_multi_head_attention_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_122/ReadVariableOpReadVariableOpEread_122_disablecopyonread_adam_m_multi_head_attention_2_query_kernel^Read_122/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_123/DisableCopyOnReadDisableCopyOnReadEread_123_disablecopyonread_adam_v_multi_head_attention_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_123/ReadVariableOpReadVariableOpEread_123_disablecopyonread_adam_v_multi_head_attention_2_query_kernel^Read_123/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_124/DisableCopyOnReadDisableCopyOnReadCread_124_disablecopyonread_adam_m_multi_head_attention_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_124/ReadVariableOpReadVariableOpCread_124_disablecopyonread_adam_m_multi_head_attention_2_query_bias^Read_124/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_125/DisableCopyOnReadDisableCopyOnReadCread_125_disablecopyonread_adam_v_multi_head_attention_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_125/ReadVariableOpReadVariableOpCread_125_disablecopyonread_adam_v_multi_head_attention_2_query_bias^Read_125/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_250IdentityRead_125/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_126/DisableCopyOnReadDisableCopyOnReadCread_126_disablecopyonread_adam_m_multi_head_attention_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_126/ReadVariableOpReadVariableOpCread_126_disablecopyonread_adam_m_multi_head_attention_2_key_kernel^Read_126/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_252IdentityRead_126/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_253IdentityIdentity_252:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_127/DisableCopyOnReadDisableCopyOnReadCread_127_disablecopyonread_adam_v_multi_head_attention_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_127/ReadVariableOpReadVariableOpCread_127_disablecopyonread_adam_v_multi_head_attention_2_key_kernel^Read_127/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_254IdentityRead_127/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_255IdentityIdentity_254:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_128/DisableCopyOnReadDisableCopyOnReadAread_128_disablecopyonread_adam_m_multi_head_attention_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_128/ReadVariableOpReadVariableOpAread_128_disablecopyonread_adam_m_multi_head_attention_2_key_bias^Read_128/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_256IdentityRead_128/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_257IdentityIdentity_256:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_129/DisableCopyOnReadDisableCopyOnReadAread_129_disablecopyonread_adam_v_multi_head_attention_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_129/ReadVariableOpReadVariableOpAread_129_disablecopyonread_adam_v_multi_head_attention_2_key_bias^Read_129/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_258IdentityRead_129/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_259IdentityIdentity_258:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_130/DisableCopyOnReadDisableCopyOnReadEread_130_disablecopyonread_adam_m_multi_head_attention_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_130/ReadVariableOpReadVariableOpEread_130_disablecopyonread_adam_m_multi_head_attention_2_value_kernel^Read_130/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_260IdentityRead_130/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_261IdentityIdentity_260:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_131/DisableCopyOnReadDisableCopyOnReadEread_131_disablecopyonread_adam_v_multi_head_attention_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_131/ReadVariableOpReadVariableOpEread_131_disablecopyonread_adam_v_multi_head_attention_2_value_kernel^Read_131/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_262IdentityRead_131/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_263IdentityIdentity_262:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_132/DisableCopyOnReadDisableCopyOnReadCread_132_disablecopyonread_adam_m_multi_head_attention_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_132/ReadVariableOpReadVariableOpCread_132_disablecopyonread_adam_m_multi_head_attention_2_value_bias^Read_132/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_264IdentityRead_132/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_265IdentityIdentity_264:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_133/DisableCopyOnReadDisableCopyOnReadCread_133_disablecopyonread_adam_v_multi_head_attention_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_133/ReadVariableOpReadVariableOpCread_133_disablecopyonread_adam_v_multi_head_attention_2_value_bias^Read_133/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_266IdentityRead_133/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_267IdentityIdentity_266:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_134/DisableCopyOnReadDisableCopyOnReadPread_134_disablecopyonread_adam_m_multi_head_attention_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_134/ReadVariableOpReadVariableOpPread_134_disablecopyonread_adam_m_multi_head_attention_2_attention_output_kernel^Read_134/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_268IdentityRead_134/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_269IdentityIdentity_268:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_135/DisableCopyOnReadDisableCopyOnReadPread_135_disablecopyonread_adam_v_multi_head_attention_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_135/ReadVariableOpReadVariableOpPread_135_disablecopyonread_adam_v_multi_head_attention_2_attention_output_kernel^Read_135/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_270IdentityRead_135/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_271IdentityIdentity_270:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_136/DisableCopyOnReadDisableCopyOnReadNread_136_disablecopyonread_adam_m_multi_head_attention_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_136/ReadVariableOpReadVariableOpNread_136_disablecopyonread_adam_m_multi_head_attention_2_attention_output_bias^Read_136/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_272IdentityRead_136/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_273IdentityIdentity_272:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_137/DisableCopyOnReadDisableCopyOnReadNread_137_disablecopyonread_adam_v_multi_head_attention_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_137/ReadVariableOpReadVariableOpNread_137_disablecopyonread_adam_v_multi_head_attention_2_attention_output_bias^Read_137/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_274IdentityRead_137/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_275IdentityIdentity_274:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_138/DisableCopyOnReadDisableCopyOnRead=read_138_disablecopyonread_adam_m_layer_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_138/ReadVariableOpReadVariableOp=read_138_disablecopyonread_adam_m_layer_normalization_5_gamma^Read_138/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_276IdentityRead_138/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_277IdentityIdentity_276:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_139/DisableCopyOnReadDisableCopyOnRead=read_139_disablecopyonread_adam_v_layer_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_139/ReadVariableOpReadVariableOp=read_139_disablecopyonread_adam_v_layer_normalization_5_gamma^Read_139/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_278IdentityRead_139/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_279IdentityIdentity_278:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_140/DisableCopyOnReadDisableCopyOnRead<read_140_disablecopyonread_adam_m_layer_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_140/ReadVariableOpReadVariableOp<read_140_disablecopyonread_adam_m_layer_normalization_5_beta^Read_140/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_280IdentityRead_140/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_281IdentityIdentity_280:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_141/DisableCopyOnReadDisableCopyOnRead<read_141_disablecopyonread_adam_v_layer_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_141/ReadVariableOpReadVariableOp<read_141_disablecopyonread_adam_v_layer_normalization_5_beta^Read_141/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_282IdentityRead_141/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_283IdentityIdentity_282:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_142/DisableCopyOnReadDisableCopyOnRead1read_142_disablecopyonread_adam_m_conv1d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_142/ReadVariableOpReadVariableOp1read_142_disablecopyonread_adam_m_conv1d_2_kernel^Read_142/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_284IdentityRead_142/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_285IdentityIdentity_284:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_143/DisableCopyOnReadDisableCopyOnRead1read_143_disablecopyonread_adam_v_conv1d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_143/ReadVariableOpReadVariableOp1read_143_disablecopyonread_adam_v_conv1d_2_kernel^Read_143/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_286IdentityRead_143/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_287IdentityIdentity_286:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_144/DisableCopyOnReadDisableCopyOnRead/read_144_disablecopyonread_adam_m_conv1d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_144/ReadVariableOpReadVariableOp/read_144_disablecopyonread_adam_m_conv1d_2_bias^Read_144/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_288IdentityRead_144/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_289IdentityIdentity_288:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_145/DisableCopyOnReadDisableCopyOnRead/read_145_disablecopyonread_adam_v_conv1d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_145/ReadVariableOpReadVariableOp/read_145_disablecopyonread_adam_v_conv1d_2_bias^Read_145/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_290IdentityRead_145/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_291IdentityIdentity_290:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_146/DisableCopyOnReadDisableCopyOnRead=read_146_disablecopyonread_adam_m_layer_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_146/ReadVariableOpReadVariableOp=read_146_disablecopyonread_adam_m_layer_normalization_6_gamma^Read_146/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_292IdentityRead_146/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_293IdentityIdentity_292:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_147/DisableCopyOnReadDisableCopyOnRead=read_147_disablecopyonread_adam_v_layer_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_147/ReadVariableOpReadVariableOp=read_147_disablecopyonread_adam_v_layer_normalization_6_gamma^Read_147/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_294IdentityRead_147/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_295IdentityIdentity_294:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_148/DisableCopyOnReadDisableCopyOnRead<read_148_disablecopyonread_adam_m_layer_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_148/ReadVariableOpReadVariableOp<read_148_disablecopyonread_adam_m_layer_normalization_6_beta^Read_148/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_296IdentityRead_148/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_297IdentityIdentity_296:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_149/DisableCopyOnReadDisableCopyOnRead<read_149_disablecopyonread_adam_v_layer_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_149/ReadVariableOpReadVariableOp<read_149_disablecopyonread_adam_v_layer_normalization_6_beta^Read_149/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_298IdentityRead_149/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_299IdentityIdentity_298:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_150/DisableCopyOnReadDisableCopyOnReadEread_150_disablecopyonread_adam_m_multi_head_attention_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_150/ReadVariableOpReadVariableOpEread_150_disablecopyonread_adam_m_multi_head_attention_3_query_kernel^Read_150/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_300IdentityRead_150/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_301IdentityIdentity_300:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_151/DisableCopyOnReadDisableCopyOnReadEread_151_disablecopyonread_adam_v_multi_head_attention_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_151/ReadVariableOpReadVariableOpEread_151_disablecopyonread_adam_v_multi_head_attention_3_query_kernel^Read_151/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_302IdentityRead_151/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_303IdentityIdentity_302:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_152/DisableCopyOnReadDisableCopyOnReadCread_152_disablecopyonread_adam_m_multi_head_attention_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_152/ReadVariableOpReadVariableOpCread_152_disablecopyonread_adam_m_multi_head_attention_3_query_bias^Read_152/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_304IdentityRead_152/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_305IdentityIdentity_304:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_153/DisableCopyOnReadDisableCopyOnReadCread_153_disablecopyonread_adam_v_multi_head_attention_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_153/ReadVariableOpReadVariableOpCread_153_disablecopyonread_adam_v_multi_head_attention_3_query_bias^Read_153/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_306IdentityRead_153/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_307IdentityIdentity_306:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_154/DisableCopyOnReadDisableCopyOnReadCread_154_disablecopyonread_adam_m_multi_head_attention_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_154/ReadVariableOpReadVariableOpCread_154_disablecopyonread_adam_m_multi_head_attention_3_key_kernel^Read_154/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_308IdentityRead_154/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_309IdentityIdentity_308:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_155/DisableCopyOnReadDisableCopyOnReadCread_155_disablecopyonread_adam_v_multi_head_attention_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_155/ReadVariableOpReadVariableOpCread_155_disablecopyonread_adam_v_multi_head_attention_3_key_kernel^Read_155/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_310IdentityRead_155/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_311IdentityIdentity_310:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_156/DisableCopyOnReadDisableCopyOnReadAread_156_disablecopyonread_adam_m_multi_head_attention_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_156/ReadVariableOpReadVariableOpAread_156_disablecopyonread_adam_m_multi_head_attention_3_key_bias^Read_156/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_312IdentityRead_156/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_313IdentityIdentity_312:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_157/DisableCopyOnReadDisableCopyOnReadAread_157_disablecopyonread_adam_v_multi_head_attention_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_157/ReadVariableOpReadVariableOpAread_157_disablecopyonread_adam_v_multi_head_attention_3_key_bias^Read_157/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_314IdentityRead_157/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_315IdentityIdentity_314:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_158/DisableCopyOnReadDisableCopyOnReadEread_158_disablecopyonread_adam_m_multi_head_attention_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_158/ReadVariableOpReadVariableOpEread_158_disablecopyonread_adam_m_multi_head_attention_3_value_kernel^Read_158/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_316IdentityRead_158/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_317IdentityIdentity_316:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_159/DisableCopyOnReadDisableCopyOnReadEread_159_disablecopyonread_adam_v_multi_head_attention_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_159/ReadVariableOpReadVariableOpEread_159_disablecopyonread_adam_v_multi_head_attention_3_value_kernel^Read_159/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_318IdentityRead_159/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_319IdentityIdentity_318:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_160/DisableCopyOnReadDisableCopyOnReadCread_160_disablecopyonread_adam_m_multi_head_attention_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_160/ReadVariableOpReadVariableOpCread_160_disablecopyonread_adam_m_multi_head_attention_3_value_bias^Read_160/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_320IdentityRead_160/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_321IdentityIdentity_320:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_161/DisableCopyOnReadDisableCopyOnReadCread_161_disablecopyonread_adam_v_multi_head_attention_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_161/ReadVariableOpReadVariableOpCread_161_disablecopyonread_adam_v_multi_head_attention_3_value_bias^Read_161/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0q
Identity_322IdentityRead_161/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_323IdentityIdentity_322:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_162/DisableCopyOnReadDisableCopyOnReadPread_162_disablecopyonread_adam_m_multi_head_attention_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_162/ReadVariableOpReadVariableOpPread_162_disablecopyonread_adam_m_multi_head_attention_3_attention_output_kernel^Read_162/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_324IdentityRead_162/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_325IdentityIdentity_324:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_163/DisableCopyOnReadDisableCopyOnReadPread_163_disablecopyonread_adam_v_multi_head_attention_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_163/ReadVariableOpReadVariableOpPread_163_disablecopyonread_adam_v_multi_head_attention_3_attention_output_kernel^Read_163/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@*
dtype0u
Identity_326IdentityRead_163/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@k
Identity_327IdentityIdentity_326:output:0"/device:CPU:0*
T0*"
_output_shapes
:@�
Read_164/DisableCopyOnReadDisableCopyOnReadNread_164_disablecopyonread_adam_m_multi_head_attention_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_164/ReadVariableOpReadVariableOpNread_164_disablecopyonread_adam_m_multi_head_attention_3_attention_output_bias^Read_164/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_328IdentityRead_164/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_329IdentityIdentity_328:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_165/DisableCopyOnReadDisableCopyOnReadNread_165_disablecopyonread_adam_v_multi_head_attention_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_165/ReadVariableOpReadVariableOpNread_165_disablecopyonread_adam_v_multi_head_attention_3_attention_output_bias^Read_165/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_330IdentityRead_165/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_331IdentityIdentity_330:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_166/DisableCopyOnReadDisableCopyOnRead=read_166_disablecopyonread_adam_m_layer_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_166/ReadVariableOpReadVariableOp=read_166_disablecopyonread_adam_m_layer_normalization_7_gamma^Read_166/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_332IdentityRead_166/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_333IdentityIdentity_332:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_167/DisableCopyOnReadDisableCopyOnRead=read_167_disablecopyonread_adam_v_layer_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_167/ReadVariableOpReadVariableOp=read_167_disablecopyonread_adam_v_layer_normalization_7_gamma^Read_167/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_334IdentityRead_167/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_335IdentityIdentity_334:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_168/DisableCopyOnReadDisableCopyOnRead<read_168_disablecopyonread_adam_m_layer_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_168/ReadVariableOpReadVariableOp<read_168_disablecopyonread_adam_m_layer_normalization_7_beta^Read_168/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_336IdentityRead_168/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_337IdentityIdentity_336:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_169/DisableCopyOnReadDisableCopyOnRead<read_169_disablecopyonread_adam_v_layer_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_169/ReadVariableOpReadVariableOp<read_169_disablecopyonread_adam_v_layer_normalization_7_beta^Read_169/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_338IdentityRead_169/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_339IdentityIdentity_338:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_170/DisableCopyOnReadDisableCopyOnRead1read_170_disablecopyonread_adam_m_conv1d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_170/ReadVariableOpReadVariableOp1read_170_disablecopyonread_adam_m_conv1d_3_kernel^Read_170/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_340IdentityRead_170/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_341IdentityIdentity_340:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_171/DisableCopyOnReadDisableCopyOnRead1read_171_disablecopyonread_adam_v_conv1d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_171/ReadVariableOpReadVariableOp1read_171_disablecopyonread_adam_v_conv1d_3_kernel^Read_171/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0u
Identity_342IdentityRead_171/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_343IdentityIdentity_342:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_172/DisableCopyOnReadDisableCopyOnRead/read_172_disablecopyonread_adam_m_conv1d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_172/ReadVariableOpReadVariableOp/read_172_disablecopyonread_adam_m_conv1d_3_bias^Read_172/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_344IdentityRead_172/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_345IdentityIdentity_344:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_173/DisableCopyOnReadDisableCopyOnRead/read_173_disablecopyonread_adam_v_conv1d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_173/ReadVariableOpReadVariableOp/read_173_disablecopyonread_adam_v_conv1d_3_bias^Read_173/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_346IdentityRead_173/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_347IdentityIdentity_346:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_174/DisableCopyOnReadDisableCopyOnRead.read_174_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_174/ReadVariableOpReadVariableOp.read_174_disablecopyonread_adam_m_dense_kernel^Read_174/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0r
Identity_348IdentityRead_174/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_349IdentityIdentity_348:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_175/DisableCopyOnReadDisableCopyOnRead.read_175_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_175/ReadVariableOpReadVariableOp.read_175_disablecopyonread_adam_v_dense_kernel^Read_175/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0r
Identity_350IdentityRead_175/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_351IdentityIdentity_350:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_176/DisableCopyOnReadDisableCopyOnRead,read_176_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_176/ReadVariableOpReadVariableOp,read_176_disablecopyonread_adam_m_dense_bias^Read_176/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_352IdentityRead_176/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_353IdentityIdentity_352:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_177/DisableCopyOnReadDisableCopyOnRead,read_177_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_177/ReadVariableOpReadVariableOp,read_177_disablecopyonread_adam_v_dense_bias^Read_177/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_354IdentityRead_177/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_355IdentityIdentity_354:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_178/DisableCopyOnReadDisableCopyOnRead0read_178_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_178/ReadVariableOpReadVariableOp0read_178_disablecopyonread_adam_m_dense_1_kernel^Read_178/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0r
Identity_356IdentityRead_178/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_357IdentityIdentity_356:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_179/DisableCopyOnReadDisableCopyOnRead0read_179_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_179/ReadVariableOpReadVariableOp0read_179_disablecopyonread_adam_v_dense_1_kernel^Read_179/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0r
Identity_358IdentityRead_179/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_359IdentityIdentity_358:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_180/DisableCopyOnReadDisableCopyOnRead.read_180_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_180/ReadVariableOpReadVariableOp.read_180_disablecopyonread_adam_m_dense_1_bias^Read_180/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_360IdentityRead_180/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_361IdentityIdentity_360:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_181/DisableCopyOnReadDisableCopyOnRead.read_181_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_181/ReadVariableOpReadVariableOp.read_181_disablecopyonread_adam_v_dense_1_bias^Read_181/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_362IdentityRead_181/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_363IdentityIdentity_362:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_182/DisableCopyOnReadDisableCopyOnRead"read_182_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_182/ReadVariableOpReadVariableOp"read_182_disablecopyonread_total_1^Read_182/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_364IdentityRead_182/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_365IdentityIdentity_364:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_183/DisableCopyOnReadDisableCopyOnRead"read_183_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_183/ReadVariableOpReadVariableOp"read_183_disablecopyonread_count_1^Read_183/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_366IdentityRead_183/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_367IdentityIdentity_366:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_184/DisableCopyOnReadDisableCopyOnRead read_184_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_184/ReadVariableOpReadVariableOp read_184_disablecopyonread_total^Read_184/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_368IdentityRead_184/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_369IdentityIdentity_368:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_185/DisableCopyOnReadDisableCopyOnRead read_185_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_185/ReadVariableOpReadVariableOp read_185_disablecopyonread_count^Read_185/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_370IdentityRead_185/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_371IdentityIdentity_370:output:0"/device:CPU:0*
T0*
_output_shapes
: �J
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�J
value�IB�I�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/105/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/106/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/107/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/108/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/109/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/110/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/111/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/112/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/113/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/114/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/115/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/116/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/117/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/118/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/119/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/120/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0Identity_253:output:0Identity_255:output:0Identity_257:output:0Identity_259:output:0Identity_261:output:0Identity_263:output:0Identity_265:output:0Identity_267:output:0Identity_269:output:0Identity_271:output:0Identity_273:output:0Identity_275:output:0Identity_277:output:0Identity_279:output:0Identity_281:output:0Identity_283:output:0Identity_285:output:0Identity_287:output:0Identity_289:output:0Identity_291:output:0Identity_293:output:0Identity_295:output:0Identity_297:output:0Identity_299:output:0Identity_301:output:0Identity_303:output:0Identity_305:output:0Identity_307:output:0Identity_309:output:0Identity_311:output:0Identity_313:output:0Identity_315:output:0Identity_317:output:0Identity_319:output:0Identity_321:output:0Identity_323:output:0Identity_325:output:0Identity_327:output:0Identity_329:output:0Identity_331:output:0Identity_333:output:0Identity_335:output:0Identity_337:output:0Identity_339:output:0Identity_341:output:0Identity_343:output:0Identity_345:output:0Identity_347:output:0Identity_349:output:0Identity_351:output:0Identity_353:output:0Identity_355:output:0Identity_357:output:0Identity_359:output:0Identity_361:output:0Identity_363:output:0Identity_365:output:0Identity_367:output:0Identity_369:output:0Identity_371:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes�
�2�	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_372Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_373IdentityIdentity_372:output:0^NoOp*
T0*
_output_shapes
: �N
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_126/DisableCopyOnRead^Read_126/ReadVariableOp^Read_127/DisableCopyOnRead^Read_127/ReadVariableOp^Read_128/DisableCopyOnRead^Read_128/ReadVariableOp^Read_129/DisableCopyOnRead^Read_129/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_130/DisableCopyOnRead^Read_130/ReadVariableOp^Read_131/DisableCopyOnRead^Read_131/ReadVariableOp^Read_132/DisableCopyOnRead^Read_132/ReadVariableOp^Read_133/DisableCopyOnRead^Read_133/ReadVariableOp^Read_134/DisableCopyOnRead^Read_134/ReadVariableOp^Read_135/DisableCopyOnRead^Read_135/ReadVariableOp^Read_136/DisableCopyOnRead^Read_136/ReadVariableOp^Read_137/DisableCopyOnRead^Read_137/ReadVariableOp^Read_138/DisableCopyOnRead^Read_138/ReadVariableOp^Read_139/DisableCopyOnRead^Read_139/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_140/DisableCopyOnRead^Read_140/ReadVariableOp^Read_141/DisableCopyOnRead^Read_141/ReadVariableOp^Read_142/DisableCopyOnRead^Read_142/ReadVariableOp^Read_143/DisableCopyOnRead^Read_143/ReadVariableOp^Read_144/DisableCopyOnRead^Read_144/ReadVariableOp^Read_145/DisableCopyOnRead^Read_145/ReadVariableOp^Read_146/DisableCopyOnRead^Read_146/ReadVariableOp^Read_147/DisableCopyOnRead^Read_147/ReadVariableOp^Read_148/DisableCopyOnRead^Read_148/ReadVariableOp^Read_149/DisableCopyOnRead^Read_149/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_150/DisableCopyOnRead^Read_150/ReadVariableOp^Read_151/DisableCopyOnRead^Read_151/ReadVariableOp^Read_152/DisableCopyOnRead^Read_152/ReadVariableOp^Read_153/DisableCopyOnRead^Read_153/ReadVariableOp^Read_154/DisableCopyOnRead^Read_154/ReadVariableOp^Read_155/DisableCopyOnRead^Read_155/ReadVariableOp^Read_156/DisableCopyOnRead^Read_156/ReadVariableOp^Read_157/DisableCopyOnRead^Read_157/ReadVariableOp^Read_158/DisableCopyOnRead^Read_158/ReadVariableOp^Read_159/DisableCopyOnRead^Read_159/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_160/DisableCopyOnRead^Read_160/ReadVariableOp^Read_161/DisableCopyOnRead^Read_161/ReadVariableOp^Read_162/DisableCopyOnRead^Read_162/ReadVariableOp^Read_163/DisableCopyOnRead^Read_163/ReadVariableOp^Read_164/DisableCopyOnRead^Read_164/ReadVariableOp^Read_165/DisableCopyOnRead^Read_165/ReadVariableOp^Read_166/DisableCopyOnRead^Read_166/ReadVariableOp^Read_167/DisableCopyOnRead^Read_167/ReadVariableOp^Read_168/DisableCopyOnRead^Read_168/ReadVariableOp^Read_169/DisableCopyOnRead^Read_169/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_170/DisableCopyOnRead^Read_170/ReadVariableOp^Read_171/DisableCopyOnRead^Read_171/ReadVariableOp^Read_172/DisableCopyOnRead^Read_172/ReadVariableOp^Read_173/DisableCopyOnRead^Read_173/ReadVariableOp^Read_174/DisableCopyOnRead^Read_174/ReadVariableOp^Read_175/DisableCopyOnRead^Read_175/ReadVariableOp^Read_176/DisableCopyOnRead^Read_176/ReadVariableOp^Read_177/DisableCopyOnRead^Read_177/ReadVariableOp^Read_178/DisableCopyOnRead^Read_178/ReadVariableOp^Read_179/DisableCopyOnRead^Read_179/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_180/DisableCopyOnRead^Read_180/ReadVariableOp^Read_181/DisableCopyOnRead^Read_181/ReadVariableOp^Read_182/DisableCopyOnRead^Read_182/ReadVariableOp^Read_183/DisableCopyOnRead^Read_183/ReadVariableOp^Read_184/DisableCopyOnRead^Read_184/ReadVariableOp^Read_185/DisableCopyOnRead^Read_185/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_373Identity_373:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp28
Read_126/DisableCopyOnReadRead_126/DisableCopyOnRead22
Read_126/ReadVariableOpRead_126/ReadVariableOp28
Read_127/DisableCopyOnReadRead_127/DisableCopyOnRead22
Read_127/ReadVariableOpRead_127/ReadVariableOp28
Read_128/DisableCopyOnReadRead_128/DisableCopyOnRead22
Read_128/ReadVariableOpRead_128/ReadVariableOp28
Read_129/DisableCopyOnReadRead_129/DisableCopyOnRead22
Read_129/ReadVariableOpRead_129/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp28
Read_130/DisableCopyOnReadRead_130/DisableCopyOnRead22
Read_130/ReadVariableOpRead_130/ReadVariableOp28
Read_131/DisableCopyOnReadRead_131/DisableCopyOnRead22
Read_131/ReadVariableOpRead_131/ReadVariableOp28
Read_132/DisableCopyOnReadRead_132/DisableCopyOnRead22
Read_132/ReadVariableOpRead_132/ReadVariableOp28
Read_133/DisableCopyOnReadRead_133/DisableCopyOnRead22
Read_133/ReadVariableOpRead_133/ReadVariableOp28
Read_134/DisableCopyOnReadRead_134/DisableCopyOnRead22
Read_134/ReadVariableOpRead_134/ReadVariableOp28
Read_135/DisableCopyOnReadRead_135/DisableCopyOnRead22
Read_135/ReadVariableOpRead_135/ReadVariableOp28
Read_136/DisableCopyOnReadRead_136/DisableCopyOnRead22
Read_136/ReadVariableOpRead_136/ReadVariableOp28
Read_137/DisableCopyOnReadRead_137/DisableCopyOnRead22
Read_137/ReadVariableOpRead_137/ReadVariableOp28
Read_138/DisableCopyOnReadRead_138/DisableCopyOnRead22
Read_138/ReadVariableOpRead_138/ReadVariableOp28
Read_139/DisableCopyOnReadRead_139/DisableCopyOnRead22
Read_139/ReadVariableOpRead_139/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp28
Read_140/DisableCopyOnReadRead_140/DisableCopyOnRead22
Read_140/ReadVariableOpRead_140/ReadVariableOp28
Read_141/DisableCopyOnReadRead_141/DisableCopyOnRead22
Read_141/ReadVariableOpRead_141/ReadVariableOp28
Read_142/DisableCopyOnReadRead_142/DisableCopyOnRead22
Read_142/ReadVariableOpRead_142/ReadVariableOp28
Read_143/DisableCopyOnReadRead_143/DisableCopyOnRead22
Read_143/ReadVariableOpRead_143/ReadVariableOp28
Read_144/DisableCopyOnReadRead_144/DisableCopyOnRead22
Read_144/ReadVariableOpRead_144/ReadVariableOp28
Read_145/DisableCopyOnReadRead_145/DisableCopyOnRead22
Read_145/ReadVariableOpRead_145/ReadVariableOp28
Read_146/DisableCopyOnReadRead_146/DisableCopyOnRead22
Read_146/ReadVariableOpRead_146/ReadVariableOp28
Read_147/DisableCopyOnReadRead_147/DisableCopyOnRead22
Read_147/ReadVariableOpRead_147/ReadVariableOp28
Read_148/DisableCopyOnReadRead_148/DisableCopyOnRead22
Read_148/ReadVariableOpRead_148/ReadVariableOp28
Read_149/DisableCopyOnReadRead_149/DisableCopyOnRead22
Read_149/ReadVariableOpRead_149/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp28
Read_150/DisableCopyOnReadRead_150/DisableCopyOnRead22
Read_150/ReadVariableOpRead_150/ReadVariableOp28
Read_151/DisableCopyOnReadRead_151/DisableCopyOnRead22
Read_151/ReadVariableOpRead_151/ReadVariableOp28
Read_152/DisableCopyOnReadRead_152/DisableCopyOnRead22
Read_152/ReadVariableOpRead_152/ReadVariableOp28
Read_153/DisableCopyOnReadRead_153/DisableCopyOnRead22
Read_153/ReadVariableOpRead_153/ReadVariableOp28
Read_154/DisableCopyOnReadRead_154/DisableCopyOnRead22
Read_154/ReadVariableOpRead_154/ReadVariableOp28
Read_155/DisableCopyOnReadRead_155/DisableCopyOnRead22
Read_155/ReadVariableOpRead_155/ReadVariableOp28
Read_156/DisableCopyOnReadRead_156/DisableCopyOnRead22
Read_156/ReadVariableOpRead_156/ReadVariableOp28
Read_157/DisableCopyOnReadRead_157/DisableCopyOnRead22
Read_157/ReadVariableOpRead_157/ReadVariableOp28
Read_158/DisableCopyOnReadRead_158/DisableCopyOnRead22
Read_158/ReadVariableOpRead_158/ReadVariableOp28
Read_159/DisableCopyOnReadRead_159/DisableCopyOnRead22
Read_159/ReadVariableOpRead_159/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp28
Read_160/DisableCopyOnReadRead_160/DisableCopyOnRead22
Read_160/ReadVariableOpRead_160/ReadVariableOp28
Read_161/DisableCopyOnReadRead_161/DisableCopyOnRead22
Read_161/ReadVariableOpRead_161/ReadVariableOp28
Read_162/DisableCopyOnReadRead_162/DisableCopyOnRead22
Read_162/ReadVariableOpRead_162/ReadVariableOp28
Read_163/DisableCopyOnReadRead_163/DisableCopyOnRead22
Read_163/ReadVariableOpRead_163/ReadVariableOp28
Read_164/DisableCopyOnReadRead_164/DisableCopyOnRead22
Read_164/ReadVariableOpRead_164/ReadVariableOp28
Read_165/DisableCopyOnReadRead_165/DisableCopyOnRead22
Read_165/ReadVariableOpRead_165/ReadVariableOp28
Read_166/DisableCopyOnReadRead_166/DisableCopyOnRead22
Read_166/ReadVariableOpRead_166/ReadVariableOp28
Read_167/DisableCopyOnReadRead_167/DisableCopyOnRead22
Read_167/ReadVariableOpRead_167/ReadVariableOp28
Read_168/DisableCopyOnReadRead_168/DisableCopyOnRead22
Read_168/ReadVariableOpRead_168/ReadVariableOp28
Read_169/DisableCopyOnReadRead_169/DisableCopyOnRead22
Read_169/ReadVariableOpRead_169/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp28
Read_170/DisableCopyOnReadRead_170/DisableCopyOnRead22
Read_170/ReadVariableOpRead_170/ReadVariableOp28
Read_171/DisableCopyOnReadRead_171/DisableCopyOnRead22
Read_171/ReadVariableOpRead_171/ReadVariableOp28
Read_172/DisableCopyOnReadRead_172/DisableCopyOnRead22
Read_172/ReadVariableOpRead_172/ReadVariableOp28
Read_173/DisableCopyOnReadRead_173/DisableCopyOnRead22
Read_173/ReadVariableOpRead_173/ReadVariableOp28
Read_174/DisableCopyOnReadRead_174/DisableCopyOnRead22
Read_174/ReadVariableOpRead_174/ReadVariableOp28
Read_175/DisableCopyOnReadRead_175/DisableCopyOnRead22
Read_175/ReadVariableOpRead_175/ReadVariableOp28
Read_176/DisableCopyOnReadRead_176/DisableCopyOnRead22
Read_176/ReadVariableOpRead_176/ReadVariableOp28
Read_177/DisableCopyOnReadRead_177/DisableCopyOnRead22
Read_177/ReadVariableOpRead_177/ReadVariableOp28
Read_178/DisableCopyOnReadRead_178/DisableCopyOnRead22
Read_178/ReadVariableOpRead_178/ReadVariableOp28
Read_179/DisableCopyOnReadRead_179/DisableCopyOnRead22
Read_179/ReadVariableOpRead_179/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp28
Read_180/DisableCopyOnReadRead_180/DisableCopyOnRead22
Read_180/ReadVariableOpRead_180/ReadVariableOp28
Read_181/DisableCopyOnReadRead_181/DisableCopyOnRead22
Read_181/ReadVariableOpRead_181/ReadVariableOp28
Read_182/DisableCopyOnReadRead_182/DisableCopyOnRead22
Read_182/ReadVariableOpRead_182/ReadVariableOp28
Read_183/DisableCopyOnReadRead_183/DisableCopyOnRead22
Read_183/ReadVariableOpRead_183/ReadVariableOp28
Read_184/DisableCopyOnReadRead_184/DisableCopyOnRead22
Read_184/ReadVariableOpRead_184/ReadVariableOp28
Read_185/DisableCopyOnReadRead_185/DisableCopyOnRead22
Read_185/ReadVariableOpRead_185/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:>�9

_output_shapes
: 

_user_specified_nameConst:&�!

_user_specified_namecount:&�!

_user_specified_nametotal:(�#
!
_user_specified_name	count_1:(�#
!
_user_specified_name	total_1:4�/
-
_user_specified_nameAdam/v/dense_1/bias:4�/
-
_user_specified_nameAdam/m/dense_1/bias:6�1
/
_user_specified_nameAdam/v/dense_1/kernel:6�1
/
_user_specified_nameAdam/m/dense_1/kernel:2�-
+
_user_specified_nameAdam/v/dense/bias:2�-
+
_user_specified_nameAdam/m/dense/bias:4�/
-
_user_specified_nameAdam/v/dense/kernel:4�/
-
_user_specified_nameAdam/m/dense/kernel:5�0
.
_user_specified_nameAdam/v/conv1d_3/bias:5�0
.
_user_specified_nameAdam/m/conv1d_3/bias:7�2
0
_user_specified_nameAdam/v/conv1d_3/kernel:7�2
0
_user_specified_nameAdam/m/conv1d_3/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_7/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_7/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_7/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_7/gamma:T�O
M
_user_specified_name53Adam/v/multi_head_attention_3/attention_output/bias:T�O
M
_user_specified_name53Adam/m/multi_head_attention_3/attention_output/bias:V�Q
O
_user_specified_name75Adam/v/multi_head_attention_3/attention_output/kernel:V�Q
O
_user_specified_name75Adam/m/multi_head_attention_3/attention_output/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/value/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/value/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_3/value/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_3/value/kernel:G�B
@
_user_specified_name(&Adam/v/multi_head_attention_3/key/bias:G�B
@
_user_specified_name(&Adam/m/multi_head_attention_3/key/bias:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/key/kernel:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/key/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_3/query/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_3/query/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_3/query/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_3/query/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_6/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_6/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_6/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_6/gamma:5�0
.
_user_specified_nameAdam/v/conv1d_2/bias:5�0
.
_user_specified_nameAdam/m/conv1d_2/bias:7�2
0
_user_specified_nameAdam/v/conv1d_2/kernel:7�2
0
_user_specified_nameAdam/m/conv1d_2/kernel:B�=
;
_user_specified_name#!Adam/v/layer_normalization_5/beta:B�=
;
_user_specified_name#!Adam/m/layer_normalization_5/beta:C�>
<
_user_specified_name$"Adam/v/layer_normalization_5/gamma:C�>
<
_user_specified_name$"Adam/m/layer_normalization_5/gamma:T�O
M
_user_specified_name53Adam/v/multi_head_attention_2/attention_output/bias:T�O
M
_user_specified_name53Adam/m/multi_head_attention_2/attention_output/bias:V�Q
O
_user_specified_name75Adam/v/multi_head_attention_2/attention_output/kernel:V�Q
O
_user_specified_name75Adam/m/multi_head_attention_2/attention_output/kernel:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_2/value/bias:I�D
B
_user_specified_name*(Adam/m/multi_head_attention_2/value/bias:K�F
D
_user_specified_name,*Adam/v/multi_head_attention_2/value/kernel:K�F
D
_user_specified_name,*Adam/m/multi_head_attention_2/value/kernel:G�B
@
_user_specified_name(&Adam/v/multi_head_attention_2/key/bias:G�B
@
_user_specified_name(&Adam/m/multi_head_attention_2/key/bias:I�D
B
_user_specified_name*(Adam/v/multi_head_attention_2/key/kernel:HD
B
_user_specified_name*(Adam/m/multi_head_attention_2/key/kernel:H~D
B
_user_specified_name*(Adam/v/multi_head_attention_2/query/bias:H}D
B
_user_specified_name*(Adam/m/multi_head_attention_2/query/bias:J|F
D
_user_specified_name,*Adam/v/multi_head_attention_2/query/kernel:J{F
D
_user_specified_name,*Adam/m/multi_head_attention_2/query/kernel:Az=
;
_user_specified_name#!Adam/v/layer_normalization_4/beta:Ay=
;
_user_specified_name#!Adam/m/layer_normalization_4/beta:Bx>
<
_user_specified_name$"Adam/v/layer_normalization_4/gamma:Bw>
<
_user_specified_name$"Adam/m/layer_normalization_4/gamma:4v0
.
_user_specified_nameAdam/v/conv1d_1/bias:4u0
.
_user_specified_nameAdam/m/conv1d_1/bias:6t2
0
_user_specified_nameAdam/v/conv1d_1/kernel:6s2
0
_user_specified_nameAdam/m/conv1d_1/kernel:Ar=
;
_user_specified_name#!Adam/v/layer_normalization_3/beta:Aq=
;
_user_specified_name#!Adam/m/layer_normalization_3/beta:Bp>
<
_user_specified_name$"Adam/v/layer_normalization_3/gamma:Bo>
<
_user_specified_name$"Adam/m/layer_normalization_3/gamma:SnO
M
_user_specified_name53Adam/v/multi_head_attention_1/attention_output/bias:SmO
M
_user_specified_name53Adam/m/multi_head_attention_1/attention_output/bias:UlQ
O
_user_specified_name75Adam/v/multi_head_attention_1/attention_output/kernel:UkQ
O
_user_specified_name75Adam/m/multi_head_attention_1/attention_output/kernel:HjD
B
_user_specified_name*(Adam/v/multi_head_attention_1/value/bias:HiD
B
_user_specified_name*(Adam/m/multi_head_attention_1/value/bias:JhF
D
_user_specified_name,*Adam/v/multi_head_attention_1/value/kernel:JgF
D
_user_specified_name,*Adam/m/multi_head_attention_1/value/kernel:FfB
@
_user_specified_name(&Adam/v/multi_head_attention_1/key/bias:FeB
@
_user_specified_name(&Adam/m/multi_head_attention_1/key/bias:HdD
B
_user_specified_name*(Adam/v/multi_head_attention_1/key/kernel:HcD
B
_user_specified_name*(Adam/m/multi_head_attention_1/key/kernel:HbD
B
_user_specified_name*(Adam/v/multi_head_attention_1/query/bias:HaD
B
_user_specified_name*(Adam/m/multi_head_attention_1/query/bias:J`F
D
_user_specified_name,*Adam/v/multi_head_attention_1/query/kernel:J_F
D
_user_specified_name,*Adam/m/multi_head_attention_1/query/kernel:A^=
;
_user_specified_name#!Adam/v/layer_normalization_2/beta:A]=
;
_user_specified_name#!Adam/m/layer_normalization_2/beta:B\>
<
_user_specified_name$"Adam/v/layer_normalization_2/gamma:B[>
<
_user_specified_name$"Adam/m/layer_normalization_2/gamma:2Z.
,
_user_specified_nameAdam/v/conv1d/bias:2Y.
,
_user_specified_nameAdam/m/conv1d/bias:4X0
.
_user_specified_nameAdam/v/conv1d/kernel:4W0
.
_user_specified_nameAdam/m/conv1d/kernel:AV=
;
_user_specified_name#!Adam/v/layer_normalization_1/beta:AU=
;
_user_specified_name#!Adam/m/layer_normalization_1/beta:BT>
<
_user_specified_name$"Adam/v/layer_normalization_1/gamma:BS>
<
_user_specified_name$"Adam/m/layer_normalization_1/gamma:QRM
K
_user_specified_name31Adam/v/multi_head_attention/attention_output/bias:QQM
K
_user_specified_name31Adam/m/multi_head_attention/attention_output/bias:SPO
M
_user_specified_name53Adam/v/multi_head_attention/attention_output/kernel:SOO
M
_user_specified_name53Adam/m/multi_head_attention/attention_output/kernel:FNB
@
_user_specified_name(&Adam/v/multi_head_attention/value/bias:FMB
@
_user_specified_name(&Adam/m/multi_head_attention/value/bias:HLD
B
_user_specified_name*(Adam/v/multi_head_attention/value/kernel:HKD
B
_user_specified_name*(Adam/m/multi_head_attention/value/kernel:DJ@
>
_user_specified_name&$Adam/v/multi_head_attention/key/bias:DI@
>
_user_specified_name&$Adam/m/multi_head_attention/key/bias:FHB
@
_user_specified_name(&Adam/v/multi_head_attention/key/kernel:FGB
@
_user_specified_name(&Adam/m/multi_head_attention/key/kernel:FFB
@
_user_specified_name(&Adam/v/multi_head_attention/query/bias:FEB
@
_user_specified_name(&Adam/m/multi_head_attention/query/bias:HDD
B
_user_specified_name*(Adam/v/multi_head_attention/query/kernel:HCD
B
_user_specified_name*(Adam/m/multi_head_attention/query/kernel:?B;
9
_user_specified_name!Adam/v/layer_normalization/beta:?A;
9
_user_specified_name!Adam/m/layer_normalization/beta:@@<
:
_user_specified_name" Adam/v/layer_normalization/gamma:@?<
:
_user_specified_name" Adam/m/layer_normalization/gamma:->)
'
_user_specified_namelearning_rate:)=%
#
_user_specified_name	iteration:L<H
F
_user_specified_name.,multi_head_attention_3/attention_output/bias:N;J
H
_user_specified_name0.multi_head_attention_3/attention_output/kernel:A:=
;
_user_specified_name#!multi_head_attention_3/value/bias:C9?
=
_user_specified_name%#multi_head_attention_3/value/kernel:?8;
9
_user_specified_name!multi_head_attention_3/key/bias:A7=
;
_user_specified_name#!multi_head_attention_3/key/kernel:A6=
;
_user_specified_name#!multi_head_attention_3/query/bias:C5?
=
_user_specified_name%#multi_head_attention_3/query/kernel:L4H
F
_user_specified_name.,multi_head_attention_2/attention_output/bias:N3J
H
_user_specified_name0.multi_head_attention_2/attention_output/kernel:A2=
;
_user_specified_name#!multi_head_attention_2/value/bias:C1?
=
_user_specified_name%#multi_head_attention_2/value/kernel:?0;
9
_user_specified_name!multi_head_attention_2/key/bias:A/=
;
_user_specified_name#!multi_head_attention_2/key/kernel:A.=
;
_user_specified_name#!multi_head_attention_2/query/bias:C-?
=
_user_specified_name%#multi_head_attention_2/query/kernel:L,H
F
_user_specified_name.,multi_head_attention_1/attention_output/bias:N+J
H
_user_specified_name0.multi_head_attention_1/attention_output/kernel:A*=
;
_user_specified_name#!multi_head_attention_1/value/bias:C)?
=
_user_specified_name%#multi_head_attention_1/value/kernel:?(;
9
_user_specified_name!multi_head_attention_1/key/bias:A'=
;
_user_specified_name#!multi_head_attention_1/key/kernel:A&=
;
_user_specified_name#!multi_head_attention_1/query/bias:C%?
=
_user_specified_name%#multi_head_attention_1/query/kernel:J$F
D
_user_specified_name,*multi_head_attention/attention_output/bias:L#H
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?";
9
_user_specified_name!multi_head_attention/value/bias:A!=
;
_user_specified_name#!multi_head_attention/value/kernel:= 9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:-)
'
_user_specified_nameconv1d_3/bias:/+
)
_user_specified_nameconv1d_3/kernel::6
4
_user_specified_namelayer_normalization_7/beta:;7
5
_user_specified_namelayer_normalization_7/gamma::6
4
_user_specified_namelayer_normalization_6/beta:;7
5
_user_specified_namelayer_normalization_6/gamma:-)
'
_user_specified_nameconv1d_2/bias:/+
)
_user_specified_nameconv1d_2/kernel::6
4
_user_specified_namelayer_normalization_5/beta:;7
5
_user_specified_namelayer_normalization_5/gamma::6
4
_user_specified_namelayer_normalization_4/beta:;7
5
_user_specified_namelayer_normalization_4/gamma:-)
'
_user_specified_nameconv1d_1/bias:/+
)
_user_specified_nameconv1d_1/kernel::
6
4
_user_specified_namelayer_normalization_3/beta:;	7
5
_user_specified_namelayer_normalization_3/gamma::6
4
_user_specified_namelayer_normalization_2/beta:;7
5
_user_specified_namelayer_normalization_2/gamma:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv1d/kernel::6
4
_user_specified_namelayer_normalization_1/beta:;7
5
_user_specified_namelayer_normalization_1/gamma:84
2
_user_specified_namelayer_normalization/beta:95
3
_user_specified_namelayer_normalization/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_239483

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
P
$__inference_add_layer_call_fn_238876
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_237020d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_237204

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_3_layer_call_fn_239548	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237480s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239544:&"
 
_user_specified_name239542:&"
 
_user_specified_name239540:&"
 
_user_specified_name239538:&"
 
_user_specified_name239536:&"
 
_user_specified_name239534:&"
 
_user_specified_name239532:&"
 
_user_specified_name239530:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
6__inference_layer_normalization_3_layer_call_fn_239150

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_237204s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239146:&"
 
_user_specified_name239144:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237480	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
)__inference_conv1d_1_layer_call_fn_239181

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_237225s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239177:&"
 
_user_specified_name239175:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237811	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_236921

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239388	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�4
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238835	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_239812

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_237081

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
m
A__inference_add_5_layer_call_and_return_conditional_losses_239495
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_239008

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_6_layer_call_fn_239504

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_237433s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239500:&"
 
_user_specified_name239498:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239094	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_237403

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_multi_head_attention_layer_call_fn_238771	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_236997s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name238767:&"
 
_user_specified_name238765:&"
 
_user_specified_name238763:&"
 
_user_specified_name238761:&"
 
_user_specified_name238759:&"
 
_user_specified_name238757:&"
 
_user_specified_name238755:&"
 
_user_specified_name238753:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_239690

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_239219

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_dropout_2_layer_call_fn_239466

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_237844d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_239172

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_1_layer_call_and_return_conditional_losses_237225

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_238943

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_237081s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
m
A__inference_add_3_layer_call_and_return_conditional_losses_239236
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
D__inference_conv1d_1_layer_call_and_return_conditional_losses_239197

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_layer_normalization_layer_call_fn_238727

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_236950s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name238723:&"
 
_user_specified_name238721:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
R
&__inference_add_4_layer_call_fn_239394
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_237342d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
D__inference_conv1d_2_layer_call_and_return_conditional_losses_237386

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_dropout_4_layer_call_fn_239795

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_237932a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_237612

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_237601

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
A__inference_add_3_layer_call_and_return_conditional_losses_237249

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
R
&__inference_add_7_layer_call_fn_239748
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_237571d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
R
&__inference_add_3_layer_call_fn_239230
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_237249d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�,
�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237886	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�,
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237736	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
R
&__inference_add_1_layer_call_fn_238971
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_237088d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�,
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_237661	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
6__inference_layer_normalization_5_layer_call_fn_239409

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_237365s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239405:&"
 
_user_specified_name239403:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
A__inference_add_5_layer_call_and_return_conditional_losses_237410

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_dropout_3_layer_call_fn_239725

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_237919d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_239431

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_237043

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_239807

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
A__inference_add_4_layer_call_and_return_conditional_losses_237342

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_237940
input_1(
layer_normalization_237622:(
layer_normalization_237624:1
multi_head_attention_237662:@-
multi_head_attention_237664:@1
multi_head_attention_237666:@-
multi_head_attention_237668:@1
multi_head_attention_237670:@-
multi_head_attention_237672:@1
multi_head_attention_237674:@)
multi_head_attention_237676:*
layer_normalization_1_237680:*
layer_normalization_1_237682:#
conv1d_237685:
conv1d_237687:*
layer_normalization_2_237697:*
layer_normalization_2_237699:3
multi_head_attention_1_237737:@/
multi_head_attention_1_237739:@3
multi_head_attention_1_237741:@/
multi_head_attention_1_237743:@3
multi_head_attention_1_237745:@/
multi_head_attention_1_237747:@3
multi_head_attention_1_237749:@+
multi_head_attention_1_237751:*
layer_normalization_3_237755:*
layer_normalization_3_237757:%
conv1d_1_237760:
conv1d_1_237762:*
layer_normalization_4_237772:*
layer_normalization_4_237774:3
multi_head_attention_2_237812:@/
multi_head_attention_2_237814:@3
multi_head_attention_2_237816:@/
multi_head_attention_2_237818:@3
multi_head_attention_2_237820:@/
multi_head_attention_2_237822:@3
multi_head_attention_2_237824:@+
multi_head_attention_2_237826:*
layer_normalization_5_237830:*
layer_normalization_5_237832:%
conv1d_2_237835:
conv1d_2_237837:*
layer_normalization_6_237847:*
layer_normalization_6_237849:3
multi_head_attention_3_237887:@/
multi_head_attention_3_237889:@3
multi_head_attention_3_237891:@/
multi_head_attention_3_237893:@3
multi_head_attention_3_237895:@/
multi_head_attention_3_237897:@3
multi_head_attention_3_237899:@+
multi_head_attention_3_237901:*
layer_normalization_7_237905:*
layer_normalization_7_237907:%
conv1d_3_237910:
conv1d_3_237912:
dense_237923:	�
dense_237925:	�!
dense_1_237934:	�
dense_1_237936:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�.multi_head_attention_2/StatefulPartitionedCall�.multi_head_attention_3/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_237622layer_normalization_237624*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_236950�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_237662multi_head_attention_237664multi_head_attention_237666multi_head_attention_237668multi_head_attention_237670multi_head_attention_237672multi_head_attention_237674multi_head_attention_237676*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_237661�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_237020�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_237680layer_normalization_1_237682*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_237043�
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_237685conv1d_237687*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_237064�
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_237694�
add_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_237088�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_237697layer_normalization_2_237699*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_237111�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_237737multi_head_attention_1_237739multi_head_attention_1_237741multi_head_attention_1_237743multi_head_attention_1_237745multi_head_attention_1_237747multi_head_attention_1_237749multi_head_attention_1_237751*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237736�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_237181�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_237755layer_normalization_3_237757*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_237204�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_1_237760conv1d_1_237762*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_237225�
dropout_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_237769�
add_3/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_237249�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_237772layer_normalization_4_237774*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_237272�
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_237812multi_head_attention_2_237814multi_head_attention_2_237816multi_head_attention_2_237818multi_head_attention_2_237820multi_head_attention_2_237822multi_head_attention_2_237824multi_head_attention_2_237826*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237811�
add_4/PartitionedCallPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_237342�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0layer_normalization_5_237830layer_normalization_5_237832*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_237365�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_2_237835conv1d_2_237837*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_237386�
dropout_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_237844�
add_5/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_237410�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0layer_normalization_6_237847layer_normalization_6_237849*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_237433�
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_237887multi_head_attention_3_237889multi_head_attention_3_237891multi_head_attention_3_237893multi_head_attention_3_237895multi_head_attention_3_237897multi_head_attention_3_237899multi_head_attention_3_237901*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237886�
add_6/PartitionedCallPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_237503�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0layer_normalization_7_237905layer_normalization_7_237907*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_237526�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_3_237910conv1d_3_237912*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_237547�
dropout_3/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_237919�
add_7/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0add_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_237571�
(global_average_pooling1d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_236921�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_237923dense_237925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_237584�
dropout_4/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_237932�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_1_237934dense_1_237936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_237612w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:&<"
 
_user_specified_name237936:&;"
 
_user_specified_name237934:&:"
 
_user_specified_name237925:&9"
 
_user_specified_name237923:&8"
 
_user_specified_name237912:&7"
 
_user_specified_name237910:&6"
 
_user_specified_name237907:&5"
 
_user_specified_name237905:&4"
 
_user_specified_name237901:&3"
 
_user_specified_name237899:&2"
 
_user_specified_name237897:&1"
 
_user_specified_name237895:&0"
 
_user_specified_name237893:&/"
 
_user_specified_name237891:&."
 
_user_specified_name237889:&-"
 
_user_specified_name237887:&,"
 
_user_specified_name237849:&+"
 
_user_specified_name237847:&*"
 
_user_specified_name237837:&)"
 
_user_specified_name237835:&("
 
_user_specified_name237832:&'"
 
_user_specified_name237830:&&"
 
_user_specified_name237826:&%"
 
_user_specified_name237824:&$"
 
_user_specified_name237822:&#"
 
_user_specified_name237820:&""
 
_user_specified_name237818:&!"
 
_user_specified_name237816:& "
 
_user_specified_name237814:&"
 
_user_specified_name237812:&"
 
_user_specified_name237774:&"
 
_user_specified_name237772:&"
 
_user_specified_name237762:&"
 
_user_specified_name237760:&"
 
_user_specified_name237757:&"
 
_user_specified_name237755:&"
 
_user_specified_name237751:&"
 
_user_specified_name237749:&"
 
_user_specified_name237747:&"
 
_user_specified_name237745:&"
 
_user_specified_name237743:&"
 
_user_specified_name237741:&"
 
_user_specified_name237739:&"
 
_user_specified_name237737:&"
 
_user_specified_name237699:&"
 
_user_specified_name237697:&"
 
_user_specified_name237687:&"
 
_user_specified_name237685:&"
 
_user_specified_name237682:&"
 
_user_specified_name237680:&
"
 
_user_specified_name237676:&	"
 
_user_specified_name237674:&"
 
_user_specified_name237672:&"
 
_user_specified_name237670:&"
 
_user_specified_name237668:&"
 
_user_specified_name237666:&"
 
_user_specified_name237664:&"
 
_user_specified_name237662:&"
 
_user_specified_name237624:&"
 
_user_specified_name237622:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
6__inference_layer_normalization_7_layer_call_fn_239668

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_237526s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name239664:&"
 
_user_specified_name239662:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_237088

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_239224

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_2_layer_call_and_return_conditional_losses_239456

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_237181

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
&__inference_model_layer_call_fn_238190
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15:@

unknown_16:@ 

unknown_17:@

unknown_18:@ 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28: 

unknown_29:@

unknown_30:@ 

unknown_31:@

unknown_32:@ 

unknown_33:@

unknown_34:@ 

unknown_35:@

unknown_36:

unknown_37:

unknown_38: 

unknown_39:

unknown_40:

unknown_41:

unknown_42: 

unknown_43:@

unknown_44:@ 

unknown_45:@

unknown_46:@ 

unknown_47:@

unknown_48:@ 

unknown_49:@

unknown_50:

unknown_51:

unknown_52: 

unknown_53:

unknown_54:

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_237940o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&<"
 
_user_specified_name238186:&;"
 
_user_specified_name238184:&:"
 
_user_specified_name238182:&9"
 
_user_specified_name238180:&8"
 
_user_specified_name238178:&7"
 
_user_specified_name238176:&6"
 
_user_specified_name238174:&5"
 
_user_specified_name238172:&4"
 
_user_specified_name238170:&3"
 
_user_specified_name238168:&2"
 
_user_specified_name238166:&1"
 
_user_specified_name238164:&0"
 
_user_specified_name238162:&/"
 
_user_specified_name238160:&."
 
_user_specified_name238158:&-"
 
_user_specified_name238156:&,"
 
_user_specified_name238154:&+"
 
_user_specified_name238152:&*"
 
_user_specified_name238150:&)"
 
_user_specified_name238148:&("
 
_user_specified_name238146:&'"
 
_user_specified_name238144:&&"
 
_user_specified_name238142:&%"
 
_user_specified_name238140:&$"
 
_user_specified_name238138:&#"
 
_user_specified_name238136:&""
 
_user_specified_name238134:&!"
 
_user_specified_name238132:& "
 
_user_specified_name238130:&"
 
_user_specified_name238128:&"
 
_user_specified_name238126:&"
 
_user_specified_name238124:&"
 
_user_specified_name238122:&"
 
_user_specified_name238120:&"
 
_user_specified_name238118:&"
 
_user_specified_name238116:&"
 
_user_specified_name238114:&"
 
_user_specified_name238112:&"
 
_user_specified_name238110:&"
 
_user_specified_name238108:&"
 
_user_specified_name238106:&"
 
_user_specified_name238104:&"
 
_user_specified_name238102:&"
 
_user_specified_name238100:&"
 
_user_specified_name238098:&"
 
_user_specified_name238096:&"
 
_user_specified_name238094:&"
 
_user_specified_name238092:&"
 
_user_specified_name238090:&"
 
_user_specified_name238088:&
"
 
_user_specified_name238086:&	"
 
_user_specified_name238084:&"
 
_user_specified_name238082:&"
 
_user_specified_name238080:&"
 
_user_specified_name238078:&"
 
_user_specified_name238076:&"
 
_user_specified_name238074:&"
 
_user_specified_name238072:&"
 
_user_specified_name238070:&"
 
_user_specified_name238068:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_237694

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_dropout_1_layer_call_fn_239207

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_237769d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_237319	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
D__inference_conv1d_3_layer_call_and_return_conditional_losses_239715

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_237433

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_1_layer_call_fn_239202

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_237242s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_4_layer_call_fn_239790

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_237601p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_237844

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
&__inference_model_layer_call_fn_238065
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15:@

unknown_16:@ 

unknown_17:@

unknown_18:@ 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28: 

unknown_29:@

unknown_30:@ 

unknown_31:@

unknown_32:@ 

unknown_33:@

unknown_34:@ 

unknown_35:@

unknown_36:

unknown_37:

unknown_38: 

unknown_39:

unknown_40:

unknown_41:

unknown_42: 

unknown_43:@

unknown_44:@ 

unknown_45:@

unknown_46:@ 

unknown_47:@

unknown_48:@ 

unknown_49:@

unknown_50:

unknown_51:

unknown_52: 

unknown_53:

unknown_54:

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_237619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&<"
 
_user_specified_name238061:&;"
 
_user_specified_name238059:&:"
 
_user_specified_name238057:&9"
 
_user_specified_name238055:&8"
 
_user_specified_name238053:&7"
 
_user_specified_name238051:&6"
 
_user_specified_name238049:&5"
 
_user_specified_name238047:&4"
 
_user_specified_name238045:&3"
 
_user_specified_name238043:&2"
 
_user_specified_name238041:&1"
 
_user_specified_name238039:&0"
 
_user_specified_name238037:&/"
 
_user_specified_name238035:&."
 
_user_specified_name238033:&-"
 
_user_specified_name238031:&,"
 
_user_specified_name238029:&+"
 
_user_specified_name238027:&*"
 
_user_specified_name238025:&)"
 
_user_specified_name238023:&("
 
_user_specified_name238021:&'"
 
_user_specified_name238019:&&"
 
_user_specified_name238017:&%"
 
_user_specified_name238015:&$"
 
_user_specified_name238013:&#"
 
_user_specified_name238011:&""
 
_user_specified_name238009:&!"
 
_user_specified_name238007:& "
 
_user_specified_name238005:&"
 
_user_specified_name238003:&"
 
_user_specified_name238001:&"
 
_user_specified_name237999:&"
 
_user_specified_name237997:&"
 
_user_specified_name237995:&"
 
_user_specified_name237993:&"
 
_user_specified_name237991:&"
 
_user_specified_name237989:&"
 
_user_specified_name237987:&"
 
_user_specified_name237985:&"
 
_user_specified_name237983:&"
 
_user_specified_name237981:&"
 
_user_specified_name237979:&"
 
_user_specified_name237977:&"
 
_user_specified_name237975:&"
 
_user_specified_name237973:&"
 
_user_specified_name237971:&"
 
_user_specified_name237969:&"
 
_user_specified_name237967:&"
 
_user_specified_name237965:&"
 
_user_specified_name237963:&
"
 
_user_specified_name237961:&	"
 
_user_specified_name237959:&"
 
_user_specified_name237957:&"
 
_user_specified_name237955:&"
 
_user_specified_name237953:&"
 
_user_specified_name237951:&"
 
_user_specified_name237949:&"
 
_user_specified_name237947:&"
 
_user_specified_name237945:&"
 
_user_specified_name237943:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
k
A__inference_add_6_layer_call_and_return_conditional_losses_237503

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_multi_head_attention_layer_call_fn_238793	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_237661s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name238789:&"
 
_user_specified_name238787:&"
 
_user_specified_name238785:&"
 
_user_specified_name238783:&"
 
_user_specified_name238781:&"
 
_user_specified_name238779:&"
 
_user_specified_name238777:&"
 
_user_specified_name238775:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_237365

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_237158	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@:
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
m
A__inference_add_7_layer_call_and_return_conditional_losses_239754
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�

�
A__inference_dense_layer_call_and_return_conditional_losses_237584

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_3_layer_call_fn_239570	
query	
value
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_237886s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name239566:&"
 
_user_specified_name239564:&"
 
_user_specified_name239562:&"
 
_user_specified_name239560:&"
 
_user_specified_name239558:&"
 
_user_specified_name239556:&"
 
_user_specified_name239554:&"
 
_user_specified_name239552:RN
+
_output_shapes
:���������

_user_specified_namevalue:R N
+
_output_shapes
:���������

_user_specified_namequery
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_237769

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_237020

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_238749

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
R
&__inference_add_5_layer_call_fn_239489
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_237410d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
m
A__inference_add_4_layer_call_and_return_conditional_losses_239400
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������:���������:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
D
(__inference_dropout_layer_call_fn_238948

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_237694d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv1d_layer_call_fn_238922

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_237064s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name238918:&"
 
_user_specified_name238916:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�B
!__inference__wrapped_model_236915
input_1M
?model_layer_normalization_batchnorm_mul_readvariableop_resource:I
;model_layer_normalization_batchnorm_readvariableop_resource:\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:@N
<model_multi_head_attention_query_add_readvariableop_resource:@Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:@L
:model_multi_head_attention_key_add_readvariableop_resource:@\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:@N
<model_multi_head_attention_value_add_readvariableop_resource:@g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_1_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_1_batchnorm_readvariableop_resource:N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource::
,model_conv1d_biasadd_readvariableop_resource:O
Amodel_layer_normalization_2_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_2_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_1_query_add_readvariableop_resource:@\
Fmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@N
<model_multi_head_attention_1_key_add_readvariableop_resource:@^
Hmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_1_value_add_readvariableop_resource:@i
Smodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@W
Imodel_multi_head_attention_1_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_3_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_3_batchnorm_readvariableop_resource:P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_1_biasadd_readvariableop_resource:O
Amodel_layer_normalization_4_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_4_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_2_query_add_readvariableop_resource:@\
Fmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource:@N
<model_multi_head_attention_2_key_add_readvariableop_resource:@^
Hmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_2_value_add_readvariableop_resource:@i
Smodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource:@W
Imodel_multi_head_attention_2_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_5_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_5_batchnorm_readvariableop_resource:P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_2_biasadd_readvariableop_resource:O
Amodel_layer_normalization_6_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_6_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_3_query_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_3_query_add_readvariableop_resource:@\
Fmodel_multi_head_attention_3_key_einsum_einsum_readvariableop_resource:@N
<model_multi_head_attention_3_key_add_readvariableop_resource:@^
Hmodel_multi_head_attention_3_value_einsum_einsum_readvariableop_resource:@P
>model_multi_head_attention_3_value_add_readvariableop_resource:@i
Smodel_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource:@W
Imodel_multi_head_attention_3_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_7_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_7_batchnorm_readvariableop_resource:P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_3_biasadd_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	�:
+model_dense_biasadd_readvariableop_resource:	�?
,model_dense_1_matmul_readvariableop_resource:	�;
-model_dense_1_biasadd_readvariableop_resource:
identity��#model/conv1d/BiasAdd/ReadVariableOp�/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_1/BiasAdd/ReadVariableOp�1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_2/BiasAdd/ReadVariableOp�1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_3/BiasAdd/ReadVariableOp�1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�2model/layer_normalization/batchnorm/ReadVariableOp�6model/layer_normalization/batchnorm/mul/ReadVariableOp�4model/layer_normalization_1/batchnorm/ReadVariableOp�8model/layer_normalization_1/batchnorm/mul/ReadVariableOp�4model/layer_normalization_2/batchnorm/ReadVariableOp�8model/layer_normalization_2/batchnorm/mul/ReadVariableOp�4model/layer_normalization_3/batchnorm/ReadVariableOp�8model/layer_normalization_3/batchnorm/mul/ReadVariableOp�4model/layer_normalization_4/batchnorm/ReadVariableOp�8model/layer_normalization_4/batchnorm/mul/ReadVariableOp�4model/layer_normalization_5/batchnorm/ReadVariableOp�8model/layer_normalization_5/batchnorm/mul/ReadVariableOp�4model/layer_normalization_6/batchnorm/ReadVariableOp�8model/layer_normalization_6/batchnorm/mul/ReadVariableOp�4model/layer_normalization_7/batchnorm/ReadVariableOp�8model/layer_normalization_7/batchnorm/mul/ReadVariableOp�>model/multi_head_attention/attention_output/add/ReadVariableOp�Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�1model/multi_head_attention/key/add/ReadVariableOp�;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/query/add/ReadVariableOp�=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/value/add/ReadVariableOp�=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp�@model/multi_head_attention_1/attention_output/add/ReadVariableOp�Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�3model/multi_head_attention_1/key/add/ReadVariableOp�=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_1/query/add/ReadVariableOp�?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_1/value/add/ReadVariableOp�?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�@model/multi_head_attention_2/attention_output/add/ReadVariableOp�Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp�3model/multi_head_attention_2/key/add/ReadVariableOp�=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_2/query/add/ReadVariableOp�?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_2/value/add/ReadVariableOp�?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp�@model/multi_head_attention_3/attention_output/add/ReadVariableOp�Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp�3model/multi_head_attention_3/key/add/ReadVariableOp�=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_3/query/add/ReadVariableOp�?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp�5model/multi_head_attention_3/value/add/ReadVariableOp�?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp�
8model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&model/layer_normalization/moments/meanMeaninput_1Amodel/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:����������
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_17model/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(n
)model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
6model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0>model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
)model/layer_normalization/batchnorm/mul_1Mulinput_1+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
2model/layer_normalization/batchnorm/ReadVariableOpReadVariableOp;model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
'model/layer_normalization/batchnorm/subSub:model/layer_normalization/batchnorm/ReadVariableOp:value:0-model/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:����������
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
.model/multi_head_attention/query/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
,model/multi_head_attention/key/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
.model/multi_head_attention/value/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
model/add/addAddV23model/multi_head_attention/attention_output/add:z:0input_1*
T0*+
_output_shapes
:����������
:model/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_1/moments/meanMeanmodel/add/add:z:0Cmodel/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_1/moments/StopGradientStopGradient1model/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencemodel/add/add:z:09model/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_1/moments/varianceMean9model/layer_normalization_1/moments/SquaredDifference:z:0Gmodel/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_1/batchnorm/addAddV25model/layer_normalization_1/moments/variance:output:04model/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_1/batchnorm/RsqrtRsqrt-model/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_1/batchnorm/mulMul/model/layer_normalization_1/batchnorm/Rsqrt:y:0@model/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_1/batchnorm/mul_1Mulmodel/add/add:z:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_1/batchnorm/mul_2Mul1model/layer_normalization_1/moments/mean:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_1/batchnorm/subSub<model/layer_normalization_1/batchnorm/ReadVariableOp:value:0/model/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_1/batchnorm/add_1AddV2/model/layer_normalization_1/batchnorm/mul_1:z:0-model/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������m
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/conv1d/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_1/batchnorm/add_1:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������n
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������y
model/dropout/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*+
_output_shapes
:����������
model/add_1/addAddV2model/dropout/Identity:output:0model/add/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_2/moments/meanMeanmodel/add_1/add:z:0Cmodel/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_2/moments/StopGradientStopGradient1model/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencemodel/add_1/add:z:09model/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_2/moments/varianceMean9model/layer_normalization_2/moments/SquaredDifference:z:0Gmodel/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_2/batchnorm/addAddV25model/layer_normalization_2/moments/variance:output:04model/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_2/batchnorm/RsqrtRsqrt-model/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_2/batchnorm/mulMul/model/layer_normalization_2/batchnorm/Rsqrt:y:0@model/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_2/batchnorm/mul_1Mulmodel/add_1/add:z:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_2/batchnorm/mul_2Mul1model/layer_normalization_2/moments/mean:output:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_2/batchnorm/subSub<model/layer_normalization_2/batchnorm/ReadVariableOp:value:0/model/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_2/batchnorm/add_1AddV2/model/layer_normalization_2/batchnorm/mul_1:z:0-model/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:����������
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_1/query/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Gmodel/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_1/query/addAddV29model/multi_head_attention_1/query/einsum/Einsum:output:0=model/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
.model/multi_head_attention_1/key/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Emodel/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention_1/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model/multi_head_attention_1/key/addAddV27model/multi_head_attention_1/key/einsum/Einsum:output:0;model/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_1/value/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Gmodel/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_1/value/addAddV29model/multi_head_attention_1/value/einsum/Einsum:output:0=model/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
"model/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
 model/multi_head_attention_1/MulMul*model/multi_head_attention_1/query/add:z:0+model/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
*model/multi_head_attention_1/einsum/EinsumEinsum(model/multi_head_attention_1/key/add:z:0$model/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
,model/multi_head_attention_1/softmax/SoftmaxSoftmax3model/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
-model/multi_head_attention_1/dropout/IdentityIdentity6model/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
,model/multi_head_attention_1/einsum_1/EinsumEinsum6model/multi_head_attention_1/dropout/Identity:output:0*model/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
;model/multi_head_attention_1/attention_output/einsum/EinsumEinsum5model/multi_head_attention_1/einsum_1/Einsum:output:0Rmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
@model/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
1model/multi_head_attention_1/attention_output/addAddV2Dmodel/multi_head_attention_1/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
model/add_2/addAddV25model/multi_head_attention_1/attention_output/add:z:0model/add_1/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_3/moments/meanMeanmodel/add_2/add:z:0Cmodel/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_3/moments/StopGradientStopGradient1model/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencemodel/add_2/add:z:09model/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_3/moments/varianceMean9model/layer_normalization_3/moments/SquaredDifference:z:0Gmodel/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_3/batchnorm/addAddV25model/layer_normalization_3/moments/variance:output:04model/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_3/batchnorm/RsqrtRsqrt-model/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_3/batchnorm/mulMul/model/layer_normalization_3/batchnorm/Rsqrt:y:0@model/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_3/batchnorm/mul_1Mulmodel/add_2/add:z:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_3/batchnorm/mul_2Mul1model/layer_normalization_3/moments/mean:output:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_3/batchnorm/subSub<model/layer_normalization_3/batchnorm/ReadVariableOp:value:0/model/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_3/batchnorm/add_1AddV2/model/layer_normalization_3/batchnorm/mul_1:z:0-model/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������o
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_1/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_3/batchnorm/add_1:z:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������r
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������}
model/dropout_1/IdentityIdentity!model/conv1d_1/Relu:activations:0*
T0*+
_output_shapes
:����������
model/add_3/addAddV2!model/dropout_1/Identity:output:0model/add_2/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_4/moments/meanMeanmodel/add_3/add:z:0Cmodel/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_4/moments/StopGradientStopGradient1model/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencemodel/add_3/add:z:09model/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_4/moments/varianceMean9model/layer_normalization_4/moments/SquaredDifference:z:0Gmodel/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_4/batchnorm/addAddV25model/layer_normalization_4/moments/variance:output:04model/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_4/batchnorm/RsqrtRsqrt-model/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_4/batchnorm/mulMul/model/layer_normalization_4/batchnorm/Rsqrt:y:0@model/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_4/batchnorm/mul_1Mulmodel/add_3/add:z:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_4/batchnorm/mul_2Mul1model/layer_normalization_4/moments/mean:output:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_4/batchnorm/subSub<model/layer_normalization_4/batchnorm/ReadVariableOp:value:0/model/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_4/batchnorm/add_1AddV2/model/layer_normalization_4/batchnorm/mul_1:z:0-model/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:����������
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_2/query/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Gmodel/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_2/query/addAddV29model/multi_head_attention_2/query/einsum/Einsum:output:0=model/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
.model/multi_head_attention_2/key/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Emodel/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention_2/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model/multi_head_attention_2/key/addAddV27model/multi_head_attention_2/key/einsum/Einsum:output:0;model/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_2/value/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Gmodel/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_2/value/addAddV29model/multi_head_attention_2/value/einsum/Einsum:output:0=model/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
"model/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
 model/multi_head_attention_2/MulMul*model/multi_head_attention_2/query/add:z:0+model/multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
*model/multi_head_attention_2/einsum/EinsumEinsum(model/multi_head_attention_2/key/add:z:0$model/multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
,model/multi_head_attention_2/softmax/SoftmaxSoftmax3model/multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
-model/multi_head_attention_2/dropout/IdentityIdentity6model/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
,model/multi_head_attention_2/einsum_1/EinsumEinsum6model/multi_head_attention_2/dropout/Identity:output:0*model/multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
;model/multi_head_attention_2/attention_output/einsum/EinsumEinsum5model/multi_head_attention_2/einsum_1/Einsum:output:0Rmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
@model/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
1model/multi_head_attention_2/attention_output/addAddV2Dmodel/multi_head_attention_2/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
model/add_4/addAddV25model/multi_head_attention_2/attention_output/add:z:0model/add_3/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_5/moments/meanMeanmodel/add_4/add:z:0Cmodel/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_5/moments/StopGradientStopGradient1model/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencemodel/add_4/add:z:09model/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_5/moments/varianceMean9model/layer_normalization_5/moments/SquaredDifference:z:0Gmodel/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_5/batchnorm/addAddV25model/layer_normalization_5/moments/variance:output:04model/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_5/batchnorm/RsqrtRsqrt-model/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_5/batchnorm/mulMul/model/layer_normalization_5/batchnorm/Rsqrt:y:0@model/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_5/batchnorm/mul_1Mulmodel/add_4/add:z:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_5/batchnorm/mul_2Mul1model/layer_normalization_5/moments/mean:output:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_5/batchnorm/subSub<model/layer_normalization_5/batchnorm/ReadVariableOp:value:0/model/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_5/batchnorm/add_1AddV2/model/layer_normalization_5/batchnorm/mul_1:z:0-model/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������o
$model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_2/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_5/batchnorm/add_1:z:0-model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/conv1d_2/Conv1DConv2D)model/conv1d_2/Conv1D/ExpandDims:output:0+model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
model/conv1d_2/Conv1D/SqueezeSqueezemodel/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/Conv1D/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������r
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������}
model/dropout_2/IdentityIdentity!model/conv1d_2/Relu:activations:0*
T0*+
_output_shapes
:����������
model/add_5/addAddV2!model/dropout_2/Identity:output:0model/add_4/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_6/moments/meanMeanmodel/add_5/add:z:0Cmodel/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_6/moments/StopGradientStopGradient1model/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencemodel/add_5/add:z:09model/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_6/moments/varianceMean9model/layer_normalization_6/moments/SquaredDifference:z:0Gmodel/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_6/batchnorm/addAddV25model/layer_normalization_6/moments/variance:output:04model/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_6/batchnorm/RsqrtRsqrt-model/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_6/batchnorm/mulMul/model/layer_normalization_6/batchnorm/Rsqrt:y:0@model/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_6/batchnorm/mul_1Mulmodel/add_5/add:z:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_6/batchnorm/mul_2Mul1model/layer_normalization_6/moments/mean:output:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_6/batchnorm/subSub<model/layer_normalization_6/batchnorm/ReadVariableOp:value:0/model/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_6/batchnorm/add_1AddV2/model/layer_normalization_6/batchnorm/mul_1:z:0-model/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:����������
?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_3/query/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Gmodel/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_3/query/addAddV29model/multi_head_attention_3/query/einsum/Einsum:output:0=model/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
.model/multi_head_attention_3/key/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Emodel/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention_3/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model/multi_head_attention_3/key/addAddV27model/multi_head_attention_3/key/einsum/Einsum:output:0;model/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
0model/multi_head_attention_3/value/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Gmodel/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
5model/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention_3/value/addAddV29model/multi_head_attention_3/value/einsum/Einsum:output:0=model/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
"model/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
 model/multi_head_attention_3/MulMul*model/multi_head_attention_3/query/add:z:0+model/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
*model/multi_head_attention_3/einsum/EinsumEinsum(model/multi_head_attention_3/key/add:z:0$model/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
,model/multi_head_attention_3/softmax/SoftmaxSoftmax3model/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
-model/multi_head_attention_3/dropout/IdentityIdentity6model/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
,model/multi_head_attention_3/einsum_1/EinsumEinsum6model/multi_head_attention_3/dropout/Identity:output:0*model/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@*
dtype0�
;model/multi_head_attention_3/attention_output/einsum/EinsumEinsum5model/multi_head_attention_3/einsum_1/Einsum:output:0Rmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������*
equationabcd,cde->abe�
@model/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
1model/multi_head_attention_3/attention_output/addAddV2Dmodel/multi_head_attention_3/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
model/add_6/addAddV25model/multi_head_attention_3/attention_output/add:z:0model/add_5/add:z:0*
T0*+
_output_shapes
:����������
:model/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(model/layer_normalization_7/moments/meanMeanmodel/add_6/add:z:0Cmodel/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(�
0model/layer_normalization_7/moments/StopGradientStopGradient1model/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:����������
5model/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencemodel/add_6/add:z:09model/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:����������
>model/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,model/layer_normalization_7/moments/varianceMean9model/layer_normalization_7/moments/SquaredDifference:z:0Gmodel/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(p
+model/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
)model/layer_normalization_7/batchnorm/addAddV25model/layer_normalization_7/moments/variance:output:04model/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_7/batchnorm/RsqrtRsqrt-model/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:����������
8model/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_7/batchnorm/mulMul/model/layer_normalization_7/batchnorm/Rsqrt:y:0@model/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_7/batchnorm/mul_1Mulmodel/add_6/add:z:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_7/batchnorm/mul_2Mul1model/layer_normalization_7/moments/mean:output:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:����������
4model/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/layer_normalization_7/batchnorm/subSub<model/layer_normalization_7/batchnorm/ReadVariableOp:value:0/model/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:����������
+model/layer_normalization_7/batchnorm/add_1AddV2/model/layer_normalization_7/batchnorm/mul_1:z:0-model/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������o
$model/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_3/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_7/batchnorm/add_1:z:0-model/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_3/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/conv1d_3/Conv1DConv2D)model/conv1d_3/Conv1D/ExpandDims:output:0+model/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
model/conv1d_3/Conv1D/SqueezeSqueezemodel/conv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
%model/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv1d_3/BiasAddBiasAdd&model/conv1d_3/Conv1D/Squeeze:output:0-model/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������r
model/conv1d_3/ReluRelumodel/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������}
model/dropout_3/IdentityIdentity!model/conv1d_3/Relu:activations:0*
T0*+
_output_shapes
:����������
model/add_7/addAddV2!model/dropout_3/Identity:output:0model/add_6/add:z:0*
T0*+
_output_shapes
:���������w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
#model/global_average_pooling1d/MeanMeanmodel/add_7/add:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������w
model/dropout_4/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_2/BiasAdd/ReadVariableOp2^model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_3/BiasAdd/ReadVariableOp2^model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp3^model/layer_normalization/batchnorm/ReadVariableOp7^model/layer_normalization/batchnorm/mul/ReadVariableOp5^model/layer_normalization_1/batchnorm/ReadVariableOp9^model/layer_normalization_1/batchnorm/mul/ReadVariableOp5^model/layer_normalization_2/batchnorm/ReadVariableOp9^model/layer_normalization_2/batchnorm/mul/ReadVariableOp5^model/layer_normalization_3/batchnorm/ReadVariableOp9^model/layer_normalization_3/batchnorm/mul/ReadVariableOp5^model/layer_normalization_4/batchnorm/ReadVariableOp9^model/layer_normalization_4/batchnorm/mul/ReadVariableOp5^model/layer_normalization_5/batchnorm/ReadVariableOp9^model/layer_normalization_5/batchnorm/mul/ReadVariableOp5^model/layer_normalization_6/batchnorm/ReadVariableOp9^model/layer_normalization_6/batchnorm/mul/ReadVariableOp5^model/layer_normalization_7/batchnorm/ReadVariableOp9^model/layer_normalization_7/batchnorm/mul/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_1/attention_output/add/ReadVariableOpK^model/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_1/key/add/ReadVariableOp>^model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/query/add/ReadVariableOp@^model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/value/add/ReadVariableOp@^model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_2/attention_output/add/ReadVariableOpK^model/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_2/key/add/ReadVariableOp>^model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/query/add/ReadVariableOp@^model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/value/add/ReadVariableOp@^model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_3/attention_output/add/ReadVariableOpK^model/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_3/key/add/ReadVariableOp>^model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_3/query/add/ReadVariableOp@^model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_3/value/add/ReadVariableOp@^model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_2/BiasAdd/ReadVariableOp%model/conv1d_2/BiasAdd/ReadVariableOp2f
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_3/BiasAdd/ReadVariableOp%model/conv1d_3/BiasAdd/ReadVariableOp2f
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2h
2model/layer_normalization/batchnorm/ReadVariableOp2model/layer_normalization/batchnorm/ReadVariableOp2p
6model/layer_normalization/batchnorm/mul/ReadVariableOp6model/layer_normalization/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_1/batchnorm/ReadVariableOp4model/layer_normalization_1/batchnorm/ReadVariableOp2t
8model/layer_normalization_1/batchnorm/mul/ReadVariableOp8model/layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_2/batchnorm/ReadVariableOp4model/layer_normalization_2/batchnorm/ReadVariableOp2t
8model/layer_normalization_2/batchnorm/mul/ReadVariableOp8model/layer_normalization_2/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_3/batchnorm/ReadVariableOp4model/layer_normalization_3/batchnorm/ReadVariableOp2t
8model/layer_normalization_3/batchnorm/mul/ReadVariableOp8model/layer_normalization_3/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_4/batchnorm/ReadVariableOp4model/layer_normalization_4/batchnorm/ReadVariableOp2t
8model/layer_normalization_4/batchnorm/mul/ReadVariableOp8model/layer_normalization_4/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_5/batchnorm/ReadVariableOp4model/layer_normalization_5/batchnorm/ReadVariableOp2t
8model/layer_normalization_5/batchnorm/mul/ReadVariableOp8model/layer_normalization_5/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_6/batchnorm/ReadVariableOp4model/layer_normalization_6/batchnorm/ReadVariableOp2t
8model/layer_normalization_6/batchnorm/mul/ReadVariableOp8model/layer_normalization_6/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_7/batchnorm/ReadVariableOp4model/layer_normalization_7/batchnorm/ReadVariableOp2t
8model/layer_normalization_7/batchnorm/mul/ReadVariableOp8model/layer_normalization_7/batchnorm/mul/ReadVariableOp2�
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
@model/multi_head_attention_1/attention_output/add/ReadVariableOp@model/multi_head_attention_1/attention_output/add/ReadVariableOp2�
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_1/key/add/ReadVariableOp3model/multi_head_attention_1/key/add/ReadVariableOp2~
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/query/add/ReadVariableOp5model/multi_head_attention_1/query/add/ReadVariableOp2�
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/value/add/ReadVariableOp5model/multi_head_attention_1/value/add/ReadVariableOp2�
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2�
@model/multi_head_attention_2/attention_output/add/ReadVariableOp@model/multi_head_attention_2/attention_output/add/ReadVariableOp2�
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_2/key/add/ReadVariableOp3model/multi_head_attention_2/key/add/ReadVariableOp2~
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/query/add/ReadVariableOp5model/multi_head_attention_2/query/add/ReadVariableOp2�
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/value/add/ReadVariableOp5model/multi_head_attention_2/value/add/ReadVariableOp2�
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2�
@model/multi_head_attention_3/attention_output/add/ReadVariableOp@model/multi_head_attention_3/attention_output/add/ReadVariableOp2�
Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_3/key/add/ReadVariableOp3model/multi_head_attention_3/key/add/ReadVariableOp2~
=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_3/query/add/ReadVariableOp5model/multi_head_attention_3/query/add/ReadVariableOp2�
?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_3/value/add/ReadVariableOp5model/multi_head_attention_3/value/add/ReadVariableOp2�
?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:(<$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_237564

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_237242

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_3_layer_call_fn_239720

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_237564s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_236950

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_default_save_signature
)	optimizer
*
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1axis
	2gamma
3beta"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_query_dense
;
_key_dense
<_value_dense
=_softmax
>_dropout_layer
?_output_dense"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_query_dense
u
_key_dense
v_value_dense
w_softmax
x_dropout_layer
y_output_dense"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
20
31
�2
�3
�4
�5
�6
�7
�8
�9
M10
N11
U12
V13
l14
m15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59"
trackable_list_wrapper
�
20
31
�2
�3
�4
�5
�6
�7
�8
�9
M10
N11
U12
V13
l14
m15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
(_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_model_layer_call_fn_238065
&__inference_model_layer_call_fn_238190�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_model_layer_call_and_return_conditional_losses_237619
A__inference_model_layer_call_and_return_conditional_losses_237940�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_236915input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_layer_normalization_layer_call_fn_238727�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_238749�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_multi_head_attention_layer_call_fn_238771
5__inference_multi_head_attention_layer_call_fn_238793�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238835
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238870�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_add_layer_call_fn_238876�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_add_layer_call_and_return_conditional_losses_238882�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_1_layer_call_fn_238891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_238913�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_1/gamma
(:&2layer_normalization_1/beta
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv1d_layer_call_fn_238922�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv1d_layer_call_and_return_conditional_losses_238938�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!2conv1d/kernel
:2conv1d/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_238943
(__inference_dropout_layer_call_fn_238948�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_238960
C__inference_dropout_layer_call_and_return_conditional_losses_238965�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_1_layer_call_fn_238971�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_1_layer_call_and_return_conditional_losses_238977�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_2_layer_call_fn_238986�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_239008�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_2/gamma
(:&2layer_normalization_2/beta
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_1_layer_call_fn_239030
7__inference_multi_head_attention_1_layer_call_fn_239052�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239094
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239129�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_2_layer_call_fn_239135�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_2_layer_call_and_return_conditional_losses_239141�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_3_layer_call_fn_239150�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_239172�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_3/gamma
(:&2layer_normalization_3/beta
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv1d_1_layer_call_fn_239181�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv1d_1_layer_call_and_return_conditional_losses_239197�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2conv1d_1/kernel
:2conv1d_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_239202
*__inference_dropout_1_layer_call_fn_239207�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_239219
E__inference_dropout_1_layer_call_and_return_conditional_losses_239224�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_3_layer_call_fn_239230�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_3_layer_call_and_return_conditional_losses_239236�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_4_layer_call_fn_239245�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_239267�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_4/gamma
(:&2layer_normalization_4/beta
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_2_layer_call_fn_239289
7__inference_multi_head_attention_2_layer_call_fn_239311�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239353
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239388�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_4_layer_call_fn_239394�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_4_layer_call_and_return_conditional_losses_239400�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_5_layer_call_fn_239409�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_239431�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_5/gamma
(:&2layer_normalization_5/beta
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv1d_2_layer_call_fn_239440�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv1d_2_layer_call_and_return_conditional_losses_239456�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2conv1d_2/kernel
:2conv1d_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_2_layer_call_fn_239461
*__inference_dropout_2_layer_call_fn_239466�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_2_layer_call_and_return_conditional_losses_239478
E__inference_dropout_2_layer_call_and_return_conditional_losses_239483�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_5_layer_call_fn_239489�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_5_layer_call_and_return_conditional_losses_239495�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_6_layer_call_fn_239504�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_239526�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_6/gamma
(:&2layer_normalization_6/beta
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_3_layer_call_fn_239548
7__inference_multi_head_attention_3_layer_call_fn_239570�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239612
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239647�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_6_layer_call_fn_239653�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_6_layer_call_and_return_conditional_losses_239659�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_7_layer_call_fn_239668�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_239690�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'2layer_normalization_7/gamma
(:&2layer_normalization_7/beta
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv1d_3_layer_call_fn_239699�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv1d_3_layer_call_and_return_conditional_losses_239715�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2conv1d_3/kernel
:2conv1d_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_3_layer_call_fn_239720
*__inference_dropout_3_layer_call_fn_239725�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_3_layer_call_and_return_conditional_losses_239737
E__inference_dropout_3_layer_call_and_return_conditional_losses_239742�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_7_layer_call_fn_239748�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_7_layer_call_and_return_conditional_losses_239754�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_global_average_pooling1d_layer_call_fn_239759�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_239765�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_239774�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_239785�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_4_layer_call_fn_239790
*__inference_dropout_4_layer_call_fn_239795�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_4_layer_call_and_return_conditional_losses_239807
E__inference_dropout_4_layer_call_and_return_conditional_losses_239812�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_239821�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_239831�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�2dense_1/kernel
:2dense_1/bias
7:5@2!multi_head_attention/query/kernel
1:/@2multi_head_attention/query/bias
5:3@2multi_head_attention/key/kernel
/:-@2multi_head_attention/key/bias
7:5@2!multi_head_attention/value/kernel
1:/@2multi_head_attention/value/bias
B:@@2,multi_head_attention/attention_output/kernel
8:62*multi_head_attention/attention_output/bias
9:7@2#multi_head_attention_1/query/kernel
3:1@2!multi_head_attention_1/query/bias
7:5@2!multi_head_attention_1/key/kernel
1:/@2multi_head_attention_1/key/bias
9:7@2#multi_head_attention_1/value/kernel
3:1@2!multi_head_attention_1/value/bias
D:B@2.multi_head_attention_1/attention_output/kernel
::82,multi_head_attention_1/attention_output/bias
9:7@2#multi_head_attention_2/query/kernel
3:1@2!multi_head_attention_2/query/bias
7:5@2!multi_head_attention_2/key/kernel
1:/@2multi_head_attention_2/key/bias
9:7@2#multi_head_attention_2/value/kernel
3:1@2!multi_head_attention_2/value/bias
D:B@2.multi_head_attention_2/attention_output/kernel
::82,multi_head_attention_2/attention_output/bias
9:7@2#multi_head_attention_3/query/kernel
3:1@2!multi_head_attention_3/query/bias
7:5@2!multi_head_attention_3/key/kernel
1:/@2multi_head_attention_3/key/bias
9:7@2#multi_head_attention_3/value/kernel
3:1@2!multi_head_attention_3/value/bias
D:B@2.multi_head_attention_3/attention_output/kernel
::82,multi_head_attention_3/attention_output/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_238065input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_238190input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_237619input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_237940input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92
�93
�94
�95
�96
�97
�98
�99
�100
�101
�102
�103
�104
�105
�106
�107
�108
�109
�110
�111
�112
�113
�114
�115
�116
�117
�118
�119
�120"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_238718input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_layer_normalization_layer_call_fn_238727inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_238749inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
J
:0
;1
<2
=3
>4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_multi_head_attention_layer_call_fn_238771queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_multi_head_attention_layer_call_fn_238793queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238835queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238870queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_add_layer_call_fn_238876inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_add_layer_call_and_return_conditional_losses_238882inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_1_layer_call_fn_238891inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_238913inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv1d_layer_call_fn_238922inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv1d_layer_call_and_return_conditional_losses_238938inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_layer_call_fn_238943inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_238948inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_238960inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_238965inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_1_layer_call_fn_238971inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_1_layer_call_and_return_conditional_losses_238977inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_2_layer_call_fn_238986inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_239008inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
J
t0
u1
v2
w3
x4
y5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_1_layer_call_fn_239030queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_1_layer_call_fn_239052queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239094queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239129queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_2_layer_call_fn_239135inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_2_layer_call_and_return_conditional_losses_239141inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_3_layer_call_fn_239150inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_239172inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv1d_1_layer_call_fn_239181inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv1d_1_layer_call_and_return_conditional_losses_239197inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_1_layer_call_fn_239202inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_1_layer_call_fn_239207inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_239219inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_239224inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_3_layer_call_fn_239230inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_3_layer_call_and_return_conditional_losses_239236inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_4_layer_call_fn_239245inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_239267inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_2_layer_call_fn_239289queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_2_layer_call_fn_239311queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239353queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239388queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_4_layer_call_fn_239394inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_4_layer_call_and_return_conditional_losses_239400inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_5_layer_call_fn_239409inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_239431inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv1d_2_layer_call_fn_239440inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv1d_2_layer_call_and_return_conditional_losses_239456inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_2_layer_call_fn_239461inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_2_layer_call_fn_239466inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_2_layer_call_and_return_conditional_losses_239478inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_2_layer_call_and_return_conditional_losses_239483inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_5_layer_call_fn_239489inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_5_layer_call_and_return_conditional_losses_239495inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_6_layer_call_fn_239504inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_239526inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_3_layer_call_fn_239548queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_3_layer_call_fn_239570queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239612queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239647queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_6_layer_call_fn_239653inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_6_layer_call_and_return_conditional_losses_239659inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_7_layer_call_fn_239668inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_239690inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv1d_3_layer_call_fn_239699inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv1d_3_layer_call_and_return_conditional_losses_239715inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_3_layer_call_fn_239720inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_3_layer_call_fn_239725inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_3_layer_call_and_return_conditional_losses_239737inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_3_layer_call_and_return_conditional_losses_239742inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_7_layer_call_fn_239748inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_7_layer_call_and_return_conditional_losses_239754inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_global_average_pooling1d_layer_call_fn_239759inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_239765inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_239774inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_239785inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_4_layer_call_fn_239790inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_4_layer_call_fn_239795inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_4_layer_call_and_return_conditional_losses_239807inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_4_layer_call_and_return_conditional_losses_239812inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_239821inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_239831inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
,:*2 Adam/m/layer_normalization/gamma
,:*2 Adam/v/layer_normalization/gamma
+:)2Adam/m/layer_normalization/beta
+:)2Adam/v/layer_normalization/beta
<::@2(Adam/m/multi_head_attention/query/kernel
<::@2(Adam/v/multi_head_attention/query/kernel
6:4@2&Adam/m/multi_head_attention/query/bias
6:4@2&Adam/v/multi_head_attention/query/bias
::8@2&Adam/m/multi_head_attention/key/kernel
::8@2&Adam/v/multi_head_attention/key/kernel
4:2@2$Adam/m/multi_head_attention/key/bias
4:2@2$Adam/v/multi_head_attention/key/bias
<::@2(Adam/m/multi_head_attention/value/kernel
<::@2(Adam/v/multi_head_attention/value/kernel
6:4@2&Adam/m/multi_head_attention/value/bias
6:4@2&Adam/v/multi_head_attention/value/bias
G:E@23Adam/m/multi_head_attention/attention_output/kernel
G:E@23Adam/v/multi_head_attention/attention_output/kernel
=:;21Adam/m/multi_head_attention/attention_output/bias
=:;21Adam/v/multi_head_attention/attention_output/bias
.:,2"Adam/m/layer_normalization_1/gamma
.:,2"Adam/v/layer_normalization_1/gamma
-:+2!Adam/m/layer_normalization_1/beta
-:+2!Adam/v/layer_normalization_1/beta
(:&2Adam/m/conv1d/kernel
(:&2Adam/v/conv1d/kernel
:2Adam/m/conv1d/bias
:2Adam/v/conv1d/bias
.:,2"Adam/m/layer_normalization_2/gamma
.:,2"Adam/v/layer_normalization_2/gamma
-:+2!Adam/m/layer_normalization_2/beta
-:+2!Adam/v/layer_normalization_2/beta
>:<@2*Adam/m/multi_head_attention_1/query/kernel
>:<@2*Adam/v/multi_head_attention_1/query/kernel
8:6@2(Adam/m/multi_head_attention_1/query/bias
8:6@2(Adam/v/multi_head_attention_1/query/bias
<::@2(Adam/m/multi_head_attention_1/key/kernel
<::@2(Adam/v/multi_head_attention_1/key/kernel
6:4@2&Adam/m/multi_head_attention_1/key/bias
6:4@2&Adam/v/multi_head_attention_1/key/bias
>:<@2*Adam/m/multi_head_attention_1/value/kernel
>:<@2*Adam/v/multi_head_attention_1/value/kernel
8:6@2(Adam/m/multi_head_attention_1/value/bias
8:6@2(Adam/v/multi_head_attention_1/value/bias
I:G@25Adam/m/multi_head_attention_1/attention_output/kernel
I:G@25Adam/v/multi_head_attention_1/attention_output/kernel
?:=23Adam/m/multi_head_attention_1/attention_output/bias
?:=23Adam/v/multi_head_attention_1/attention_output/bias
.:,2"Adam/m/layer_normalization_3/gamma
.:,2"Adam/v/layer_normalization_3/gamma
-:+2!Adam/m/layer_normalization_3/beta
-:+2!Adam/v/layer_normalization_3/beta
*:(2Adam/m/conv1d_1/kernel
*:(2Adam/v/conv1d_1/kernel
 :2Adam/m/conv1d_1/bias
 :2Adam/v/conv1d_1/bias
.:,2"Adam/m/layer_normalization_4/gamma
.:,2"Adam/v/layer_normalization_4/gamma
-:+2!Adam/m/layer_normalization_4/beta
-:+2!Adam/v/layer_normalization_4/beta
>:<@2*Adam/m/multi_head_attention_2/query/kernel
>:<@2*Adam/v/multi_head_attention_2/query/kernel
8:6@2(Adam/m/multi_head_attention_2/query/bias
8:6@2(Adam/v/multi_head_attention_2/query/bias
<::@2(Adam/m/multi_head_attention_2/key/kernel
<::@2(Adam/v/multi_head_attention_2/key/kernel
6:4@2&Adam/m/multi_head_attention_2/key/bias
6:4@2&Adam/v/multi_head_attention_2/key/bias
>:<@2*Adam/m/multi_head_attention_2/value/kernel
>:<@2*Adam/v/multi_head_attention_2/value/kernel
8:6@2(Adam/m/multi_head_attention_2/value/bias
8:6@2(Adam/v/multi_head_attention_2/value/bias
I:G@25Adam/m/multi_head_attention_2/attention_output/kernel
I:G@25Adam/v/multi_head_attention_2/attention_output/kernel
?:=23Adam/m/multi_head_attention_2/attention_output/bias
?:=23Adam/v/multi_head_attention_2/attention_output/bias
.:,2"Adam/m/layer_normalization_5/gamma
.:,2"Adam/v/layer_normalization_5/gamma
-:+2!Adam/m/layer_normalization_5/beta
-:+2!Adam/v/layer_normalization_5/beta
*:(2Adam/m/conv1d_2/kernel
*:(2Adam/v/conv1d_2/kernel
 :2Adam/m/conv1d_2/bias
 :2Adam/v/conv1d_2/bias
.:,2"Adam/m/layer_normalization_6/gamma
.:,2"Adam/v/layer_normalization_6/gamma
-:+2!Adam/m/layer_normalization_6/beta
-:+2!Adam/v/layer_normalization_6/beta
>:<@2*Adam/m/multi_head_attention_3/query/kernel
>:<@2*Adam/v/multi_head_attention_3/query/kernel
8:6@2(Adam/m/multi_head_attention_3/query/bias
8:6@2(Adam/v/multi_head_attention_3/query/bias
<::@2(Adam/m/multi_head_attention_3/key/kernel
<::@2(Adam/v/multi_head_attention_3/key/kernel
6:4@2&Adam/m/multi_head_attention_3/key/bias
6:4@2&Adam/v/multi_head_attention_3/key/bias
>:<@2*Adam/m/multi_head_attention_3/value/kernel
>:<@2*Adam/v/multi_head_attention_3/value/kernel
8:6@2(Adam/m/multi_head_attention_3/value/bias
8:6@2(Adam/v/multi_head_attention_3/value/bias
I:G@25Adam/m/multi_head_attention_3/attention_output/kernel
I:G@25Adam/v/multi_head_attention_3/attention_output/kernel
?:=23Adam/m/multi_head_attention_3/attention_output/bias
?:=23Adam/v/multi_head_attention_3/attention_output/bias
.:,2"Adam/m/layer_normalization_7/gamma
.:,2"Adam/v/layer_normalization_7/gamma
-:+2!Adam/m/layer_normalization_7/beta
-:+2!Adam/v/layer_normalization_7/beta
*:(2Adam/m/conv1d_3/kernel
*:(2Adam/v/conv1d_3/kernel
 :2Adam/m/conv1d_3/bias
 :2Adam/v/conv1d_3/bias
$:"	�2Adam/m/dense/kernel
$:"	�2Adam/v/dense/kernel
:�2Adam/m/dense/bias
:�2Adam/v/dense/bias
&:$	�2Adam/m/dense_1/kernel
&:$	�2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_236915�p23��������MNUVlm��������������������������������������������4�1
*�'
%�"
input_1���������
� "1�.
,
dense_1!�
dense_1����������
A__inference_add_1_layer_call_and_return_conditional_losses_238977�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_1_layer_call_fn_238971�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_2_layer_call_and_return_conditional_losses_239141�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_2_layer_call_fn_239135�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_3_layer_call_and_return_conditional_losses_239236�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_3_layer_call_fn_239230�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_4_layer_call_and_return_conditional_losses_239400�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_4_layer_call_fn_239394�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_5_layer_call_and_return_conditional_losses_239495�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_5_layer_call_fn_239489�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_6_layer_call_and_return_conditional_losses_239659�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_6_layer_call_fn_239653�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
A__inference_add_7_layer_call_and_return_conditional_losses_239754�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
&__inference_add_7_layer_call_fn_239748�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
?__inference_add_layer_call_and_return_conditional_losses_238882�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
$__inference_add_layer_call_fn_238876�b�_
X�U
S�P
&�#
inputs_0���������
&�#
inputs_1���������
� "%�"
unknown����������
D__inference_conv1d_1_layer_call_and_return_conditional_losses_239197m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
)__inference_conv1d_1_layer_call_fn_239181b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
D__inference_conv1d_2_layer_call_and_return_conditional_losses_239456m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
)__inference_conv1d_2_layer_call_fn_239440b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
D__inference_conv1d_3_layer_call_and_return_conditional_losses_239715m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
)__inference_conv1d_3_layer_call_fn_239699b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
B__inference_conv1d_layer_call_and_return_conditional_losses_238938kUV3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
'__inference_conv1d_layer_call_fn_238922`UV3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
C__inference_dense_1_layer_call_and_return_conditional_losses_239831f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_1_layer_call_fn_239821[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_239785f��/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_239774[��/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_239219k7�4
-�*
$�!
inputs���������
p
� "0�-
&�#
tensor_0���������
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_239224k7�4
-�*
$�!
inputs���������
p 
� "0�-
&�#
tensor_0���������
� �
*__inference_dropout_1_layer_call_fn_239202`7�4
-�*
$�!
inputs���������
p
� "%�"
unknown����������
*__inference_dropout_1_layer_call_fn_239207`7�4
-�*
$�!
inputs���������
p 
� "%�"
unknown����������
E__inference_dropout_2_layer_call_and_return_conditional_losses_239478k7�4
-�*
$�!
inputs���������
p
� "0�-
&�#
tensor_0���������
� �
E__inference_dropout_2_layer_call_and_return_conditional_losses_239483k7�4
-�*
$�!
inputs���������
p 
� "0�-
&�#
tensor_0���������
� �
*__inference_dropout_2_layer_call_fn_239461`7�4
-�*
$�!
inputs���������
p
� "%�"
unknown����������
*__inference_dropout_2_layer_call_fn_239466`7�4
-�*
$�!
inputs���������
p 
� "%�"
unknown����������
E__inference_dropout_3_layer_call_and_return_conditional_losses_239737k7�4
-�*
$�!
inputs���������
p
� "0�-
&�#
tensor_0���������
� �
E__inference_dropout_3_layer_call_and_return_conditional_losses_239742k7�4
-�*
$�!
inputs���������
p 
� "0�-
&�#
tensor_0���������
� �
*__inference_dropout_3_layer_call_fn_239720`7�4
-�*
$�!
inputs���������
p
� "%�"
unknown����������
*__inference_dropout_3_layer_call_fn_239725`7�4
-�*
$�!
inputs���������
p 
� "%�"
unknown����������
E__inference_dropout_4_layer_call_and_return_conditional_losses_239807e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_4_layer_call_and_return_conditional_losses_239812e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_4_layer_call_fn_239790Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_4_layer_call_fn_239795Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
C__inference_dropout_layer_call_and_return_conditional_losses_238960k7�4
-�*
$�!
inputs���������
p
� "0�-
&�#
tensor_0���������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_238965k7�4
-�*
$�!
inputs���������
p 
� "0�-
&�#
tensor_0���������
� �
(__inference_dropout_layer_call_fn_238943`7�4
-�*
$�!
inputs���������
p
� "%�"
unknown����������
(__inference_dropout_layer_call_fn_238948`7�4
-�*
$�!
inputs���������
p 
� "%�"
unknown����������
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_239765�I�F
?�<
6�3
inputs'���������������������������

 
� "5�2
+�(
tensor_0������������������
� �
9__inference_global_average_pooling1d_layer_call_fn_239759wI�F
?�<
6�3
inputs'���������������������������

 
� "*�'
unknown�������������������
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_238913kMN3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_1_layer_call_fn_238891`MN3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_239008klm3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_2_layer_call_fn_238986`lm3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_239172m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_3_layer_call_fn_239150b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_239267m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_4_layer_call_fn_239245b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_239431m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_5_layer_call_fn_239409b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_239526m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_6_layer_call_fn_239504b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_239690m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
6__inference_layer_normalization_7_layer_call_fn_239668b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
O__inference_layer_normalization_layer_call_and_return_conditional_losses_238749k233�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
4__inference_layer_normalization_layer_call_fn_238727`233�0
)�&
$�!
inputs���������
� "%�"
unknown����������
A__inference_model_layer_call_and_return_conditional_losses_237619�p23��������MNUVlm��������������������������������������������<�9
2�/
%�"
input_1���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_237940�p23��������MNUVlm��������������������������������������������<�9
2�/
%�"
input_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
&__inference_model_layer_call_fn_238065�p23��������MNUVlm��������������������������������������������<�9
2�/
%�"
input_1���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_238190�p23��������MNUVlm��������������������������������������������<�9
2�/
%�"
input_1���������
p 

 
� "!�
unknown����������
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239094���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������
� �
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_239129���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������
� �
7__inference_multi_head_attention_1_layer_call_fn_239030���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "%�"
unknown����������
7__inference_multi_head_attention_1_layer_call_fn_239052���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "%�"
unknown����������
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239353���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������
� �
R__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_239388���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������
� �
7__inference_multi_head_attention_2_layer_call_fn_239289���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "%�"
unknown����������
7__inference_multi_head_attention_2_layer_call_fn_239311���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "%�"
unknown����������
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239612���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������
� �
R__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_239647���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������
� �
7__inference_multi_head_attention_3_layer_call_fn_239548���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "%�"
unknown����������
7__inference_multi_head_attention_3_layer_call_fn_239570���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "%�"
unknown����������
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238835���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������
� �
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_238870���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������
� �
5__inference_multi_head_attention_layer_call_fn_238771���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p
p 
� "%�"
unknown����������
5__inference_multi_head_attention_layer_call_fn_238793���������k�h
a�^
#� 
query���������
#� 
value���������

 

 
p 
p 
p 
� "%�"
unknown����������
$__inference_signature_wrapper_238718�p23��������MNUVlm��������������������������������������������?�<
� 
5�2
0
input_1%�"
input_1���������"1�.
,
dense_1!�
dense_1���������