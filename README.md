# alterbert

  参照Reformer，在Bert中引入可逆残差网络。
  
  为了计算梯度，定义了一个BPLayer和RevBPLayer，需要配合VariableScope使用。将graph切分成N个scope，这N个scope每个需要创建一个BPLayer。这N个scope不能有重叠的部分，否则重叠部分会重复计算梯度。另外修改了create_optimizer，计算梯度时调用最后一个BPLayer的backward_gradients函数，其中会递归顺序调用其他BPLayer计算梯度。

  用起来不太好用，需要仔细考虑scope的结构。
