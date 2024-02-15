# 记录Python深度学习

《Python深度学习（第2版）》

[美] 弗朗索瓦·肖莱　著
20个笔记

## 第1章 什么是深度学习

◆ 核技巧的基本思想是：要在新的表示空间中找到良好的决策超平面，不需要直接计算点在新空间中的坐标，只需要计算在新空间中点与点之间的距离，而利用核函数可以高效地完成这种计算。

◆ SVM是一种浅层方法，因此要将其应用于感知问题，首先需要手动提取出有用的表示（这一步骤叫作特征工程）。这一步骤很难，而且不稳定。如果想用SVM来进行手写数字分类，那么你不能从原始像素开始，而应该**首先手动找到有用的表示（比如前面提到的像素直方图）**，使问题变得更易于处理。

◆ 决策树：需要学习的参数是关于数据的问题

◆ 深度学习的变革之处在于，模型可以在同一时间共同学习所有表示层，而不是依次连续学习（这被称为贪婪学习

◆ 通过共同的特征学习，每当模型修改某个内部特征时，所有依赖于该特征的其他特征都会相应地自动调节适应，无须人为干预。一切都由单一反馈信号来监督：模型中的每一处变化都是为最终目标服务。这种方法比贪婪地叠加浅层模型更强大，因为它可以通过将复杂、抽象的表示拆解为多个中间空间（层）来学习这些表示，每个中间空间仅仅是前一个空间的简单变换。
深度学习从数据中进行学习时有两个基本特征：第一，通过逐层渐进的方式形成越来越复杂的表示；第二，对中间这些渐进的表示共同进行学习，每一层的修改都需要同时考虑上下两层。这两个特征叠加在一起，使得深度学习比先前的机器学习方法更成功。

◆ 要想在如今的应用机器学习中取得成功，你应该熟悉这两种技术：梯度提升树，用于浅层学习问题；深度学习，用于感知问题。

◆ 关键的问题在于通过多层叠加的梯度传播（gradient propagation）。随着层数的增加，用于训练神经网络的反馈信号会逐渐消失。

◆ 更好的优化方案（optimization scheme），比如RMSprop和Adam。
只有当这些改进让我们可以训练10层以上的模型时，深度学习才开始大放异彩。


## 第2章 神经网络的数学基础

◆ 多个3阶张量打包成一个数组，就可以创建一个4阶张量，以此类推。深度学习处理的一般是0到4阶的张量，但处理视频数据时可能会遇到5阶张量。


◆ ，在GPU上运行TensorFlow代码，逐元素运算都是通过完全向量化的CUDA来完成的，可以最大限度地利用高度并行的GPU芯片架构。


◆ SGD还有多种变体，比如带动量的SGD、Adagrad、RMSprop等。它们的不同之处在于，计算下一次权重更新时还要考虑上一次权重更新，而不是仅考虑当前的梯度值。这些变体被称为优化方法（optimization method）或优化器（optimizer）

◆ 动量的概念尤其值得关注，它被用于许多变体。动量解决了SGD的两个问题：收敛速度和局部极小值


## 第3章 Keras和TensorFlow入门

◆ 无论是在本地运行还是在云端运行，最好都使用Unix工作站。虽然从技术上来说，可以直接在Windows上运行Keras，但我并不建议这么做。如果你是Windows用户，并且想在自己的工作站上做深度学习，那么最简单的解决方案就是在你的计算机上安装Ubuntu双系统，或者利用Windows Subsystem for Linux（WSL）。WSL是一个兼容层，它让你能够在Windows上运行Linux应用程序。这可能看起来有点麻烦，但从长远来看，你可以省下大量时间并避免麻烦。

◆ tf.Variable是一个类，其作用是管理TensorFlow中的可变状态。

◆ 变量的状态可以通过其assign方法进行修改，如代码清单3-6所示。

◆ 但是NumPy无法做到的是，检索任意可微表达式相对于其输入的梯度。你只需要创建一个GradientTape作用域，对一个或多个输入张量做一些计算，然后就可以检索计算结果相对于输入的梯度，如代码清单3-10所示。

◆ 在Keras中构建模型通常有两种方法：直接作为Model类的子类，或者使用函数式API，后者可以用更少的代码做更多的事情。

◆ 一旦选定了损失函数、优化器和指标，就可以使用内置方法compile()和fit()开始训练模型。此外，也可以编写自定义的训练循环。


## 第4章 神经网络入门：分类与回归

◆ 如果某一层丢失了与分类问题相关的信息，那么后面的层永远无法恢复这些信息，也就是说，每一层都可能成为信息瓶颈。上一个例子使用了16维的中间层，但对这个例子来说，16维空间可能太小了，无法学会区分46个类别。这种维度较小的层可能成为信息瓶颈，导致相关信息永久性丢失。

◆ 如果要对N个类别的数据点进行分类，那么模型的最后一层应该是大小为N的Dense层。

    •  对于单标签、多分类问题，模型的最后一层应该使用softmax激活函数，这样可以输出一个在N个输出类别上的概率分布。
    •  对于这种问题，损失函数几乎总是应该使用分类交叉熵。它将模型输出的概率分布与目标的真实分布之间的距离最小化。
    •  处理多分类问题的标签有两种方法：
        通过分类编码（也叫one-hot编码）对标签进行编码，然后使用categorical_crossentropy损失函数；
        将标签编码为整数，然后使用sparse_categorical_crossentropy损失函数。
•  如果你需要将数据划分到多个类别中，那么应避免使用太小的中间层，以免在模型中造成信息瓶颈。


◆ 取值范围差异很大的数据输入到神经网络中，这是有问题的。模型可能会自动适应这种取值范围不同的数据，但这肯定会让学习变得更加困难。对于这类数据，普遍采用的最佳处理方法是对每个特征进行标准化，即对于输入数据的每个特征（输入数据矩阵的每一列），减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1。

◆ 注意，对测试数据进行标准化的平均值和标准差都是在训练数据上计算得到的。在深度学习工作流程中，你不能使用在测试数据上计算得到的任何结果，即使是像数据标准化这么简单的事情也不行。

◆ 一般来说，训练数据越少，过拟合就会越严重，而较小的模型可以降低过拟合。
代码清单4-25　模型定义

def build_model():
    model = keras.Sequential([  ←----由于需要将同一个模型多次实例化，因此我们用一个函数来构建模型
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

模型的最后一层只有一个单元且没有激活，它是一个线性层。这是标量回归（标量回归是预测单一连续值的回归）的典型设置。

◆ 下面是你应该从这个标量回归示例中学到的要点。
    •  回归问题使用的损失函数与分类问题不同。回归常用的损失函数是均方误差（MSE）。
    •    同样，回归问题使用的评估指标也与分类问题不同。显然，精度的概念不再适用于回归问题。常用的回归指标是平均绝对误差（MAE）。
    •  如果输入数据的特征具有不同的取值范围，那么应该先进行预处理，对每个特征单独进行缩放。
    •  如果可用的数据很少，那么K折交叉验证是评估模型的可靠方法。
    •  如果可用的训练数据很少，那么最好使用中间层较少（通常只有一两个）的小模型，以避免严重的过拟合。


◆ 对于向量数据，最常见的三类机器学习任务是：二分类问题、多分类问题和标量回归问题。
     回归问题使用的损失函数和评估指标都与分类问题不同。
    •  将原始数据输入神经网络之前，通常需要对其进行预处理。
    •  如果数据特征具有不同的取值范围，应该先进行预处理，对每个特征单独进行缩放。
    •  随着训练的进行，神经网络最终会过拟合，并在前所未见的数据上得到较差的结果。
    •  如果训练数据不是很多，那么可以使用只有一两个中间层的小模型，以避免严重的过拟合。
    •  如果数据被划分为多个类别，那么中间层过小可能会造成信息瓶颈。
    •  如果要处理的数据很少，那么K折交叉验证有助于可靠地评估模型。



## 第5章 机器学习基础

◆ 如果你不确定特征究竟是有用的还是无关紧要的，那么常见的做法是在训练前进行特征选择（feature selection）。

◆ 有用性分数（usefulness score）是用于衡量特征对于任务来说所包含信息量大小的指标，比如特征与标签之间的互信息。这么做可以过滤前面例子中的白噪声通道。

◆ 用术语来说，手写数字在28×28 uint8数组的可能性空间中构成了一个流形（manifold）。这个词看起来很高深，但其概念非常直观。“流形”是指某个父空间的低维子空间，它局部近似于一个线性空间（欧几里得空间）。例如，平面上的光滑曲线就是二维空间中的一维流形，因为对于曲线上的每一点，你都可以画出一条切线（曲线上的每一点都可以用直线来近似）。三维空间中的光滑表面是一个二维流形，以此类推。

◆ 流形假说（manifold hypothesis）假定，所有自然数据都位于高维空间中的一个低维流形中，这个高维空间是数据编码空间。

◆ 流形假说意味着：
    ◦  机器学习模型只需在其输入空间中拟合相对简单、低维、高度结构化的子空间（潜在流形）；
    ◦  在其中一个流形中，总是可以在两个输入之间进行插值（interpolate），也就是说，通过一条连续路径将一个输入变形为另一个输入，这条路径上的所有点都位于流形中。

◆ 深度学习模型不仅具有足够的表示能力，还具有以下特性，使其特别适合学习潜在流形。
    ◦  深度学习模型实现了从输入到输出的光滑连续映射。它必须是光滑连续的，因为它必须是可微的（否则无法进行梯度下降）。这种光滑性有助于逼近具有相同属性的潜在流形。
    ◦  深度学习模型的结构往往反映了训练数据中的信息“形状”（通过架构预设）.




◆ 简单的留出验证

◆ K折交叉验证

◆ 带有打乱数据的重复K折交叉验证


◆ 特征工程是指将数据输入模型之前，利用你自己关于数据和机器学习算法（这里指神经网络）的知识对数据进行硬编码的变换（这种变换不是模型学到的），以改善算法的效果。在多数情况下，机器学习模型无法从完全随意的数据中进行学习。呈现给模型的数据应该便于模型进行学习。

◆ 幸运的是，对于现代深度学习，大多数特征工程是不需要做的，因为神经网络能够从原始数据中自动提取有用的特征。这是否意味着，只要使用深度神经网络，就无须担心特征工程呢？并非如此，原因有以下两点。
    •  良好的特征仍然有助于更优雅地解决问题，同时使用更少的资源。例如，使用卷积神经网络解决读取时钟问题是非常可笑的。
    •  良好的特征可以用更少的数据解决问题。深度学习模型自主学习特征的能力依赖于拥有大量的训练数据。如果只有很少的样本，那么特征的信息价值就变得非常重要。

◆ 种最常用的正则化方法，并将其实际应用于改进第4章的影评分类模型。
1.  缩减模型容量
你已经知道，一个太小的模型不会过拟合。降低过拟合最简单的方法，就是缩减模型容量，即减少模型中可学习参数的个数（这由层数和每层单元个数决定）。
2. 添加权重正则化
你可能知道奥卡姆剃刀原理：如果一件事有两种解释，那么最可能正确的解释就是更简单的那种，即假设更少的那种。这个原理也适用于神经网络学到的模型：给定训练数据和网络架构，多组权重值（多个模型）都可以解释这些数据。简单模型比复杂模型更不容易过拟合。
这里的简单模型是指参数值分布的熵更小的模型（或参数更少的模型，比如上一节中的例子）。因此，降低过拟合的一种常见方法就是强制让模型权重只能取较小的值，从而限制模型的复杂度，这使得权重值的分布更加规则。这种方法叫作权重正则化（weight regularization），其实现方法是向模型损失函数中添加与较大权重值相关的成本（cost）。这种成本有两种形式。
    ◦  L1正则化：添加的成本与权重系数的绝对值（权重的L1范数）成正比。
    ◦  L2正则化：添加的成本与权重系数的平方（权重的L2范数）成正比。神经网络的L2正则化也叫作权重衰减（weight decay）。不要被不同的名称迷惑，权重衰减与L2正则化在数学上是完全相同的。
    权重正则化更常用于较小的深度学习模型。大型深度学习模型往往是过度参数化的，限制权重值大小对模型容量和泛化能力没有太大影响。在这种情况下，应首选另一种正则化方法：dropout。
3.  添加dropout
dropout是神经网络最常用且最有效的正则化方法之一，它由多伦多大学的Geoffrey Hinton和他的学生开发。对某一层使用dropout，就是在训练过程中随机舍弃该层的一些输出特征（将其设为0）。
    dropout的核心思想是在层的输出值中引入噪声，打破不重要的偶然模式（也就是Hinton所说的“阴谋”）。如果没有噪声，那么神经网络将记住这些偶然模式。


◆ 总结一下，要想将神经网络的泛化能力最大化，并防止过拟合，最常用的方法如下所述。
    ◦  获取更多或更好的训练数据。
    ◦  找到更好的特征。
    ◦  缩减模型容量。
    ◦  添加权重正则化（用于较小的模型）。
    ◦  添加dropout。

◆ 5.5　本章总结
•  机器学习模型的目的在于泛化，即在前所未见的输入上表现良好。这看起来不难，但实现起来很难。
•  深度神经网络实现泛化的方式是：学习一个参数化模型，这个模型可以成功地在训练样本之间进行插值——这样的模型学会了训练数据的“潜在流形”。这就是为什么深度学习模型只能理解与训练数据非常接近的输入。
•  机器学习的根本问题是优化与泛化之间的矛盾：为了实现泛化，你必须首先实现对训练数据的良好拟合，但改进模型对训练数据的拟合，在一段时间之后将不可避免地降低泛化能力。深度学习的所有最佳实践都旨在解决这一矛盾。
•  深度学习模型的泛化能力来自于这样一个事实：模型努力逼近数据的潜在流形，从而通过插值来理解新的输入。
•  在开发模型时，能够准确评估模型的泛化能力是非常重要的。你可以使用多种评估方法，包括简单的留出验证、K折交叉验证，以及带有打乱数据的重复K折交叉验证。请记住，要始终保留一个完全独立的测试集用于最终的模型评估，因为可能已经发生了从验证数据到模型的信息泄露。
•  开始构建模型时，你的目标首先是实现一个具有一定泛化能力并且能够过拟合的模型。要做到这一点，最佳做法包括调整学习率和批量大小、利用更好的架构预设、增加模型容量或者仅仅延长训练时间。
•  模型开始过拟合之后，你的目标将转为利用模型正则化来提高泛化能力。你可以缩减模型容量、添加dropout或权重正则化，以及使用EarlyStopping。当然，要想提高模型的泛化能力，首选方法始终是收集更大或更好的数据集。



## 第6章 机器学习的通用工作流程

    如果是分类特征，则可以创建一个新的类别，表示“此值缺失”。模型会自动学习这个新类别对于目标的含义。
    如果是数值特征，应避免输入像"0"这样随意的值，因为它可能会在特征形成的潜在空间中造成不连续性，从而让模型更加难以泛化。相反，你可以考虑用数据集中该特征的均值或中位值来代替缺失值。你也可以训练一个模型，给定其他特征的值，预测该特征的值。请注意，如果测试数据的分类特征有缺失值，而训练数据中没有缺失值，那么神经网络无法学会忽略缺失值。在这种情况下，你应该手动生成一些有缺失值的训练样本：将一些训练样本复制多次，然后舍弃测试数据中可能缺失的某些分类特征。

◆ 两种常用的优化方法。
    ◦  权重剪枝（weight pruning）。权重张量中的每个元素对预测结果的贡献并不相同。仅保留那些最重要的参数，可以大大减少模型的参数个数。这种方法减少了模型占用的内存资源和计算资源，而且在性能指标方面的代价很小。你可以决定剪枝的范围，从而控制模型大小与精度之间的平衡。
    ◦  权重量化（weight quantization）。深度学习模型使用单精度浮点（float32）权重进行训练。但是可以将权重量化为8位有符号整数（int8），这样得到的推断模型大小只有原始模型的四分之一，但精度仍与原始模型相当。TensorFlow生态系统包含一个权重剪枝和量化工具包，它与Keras API深度集成。



## 第7章 深入Keras

◆ 序贯模型易于使用，但适用范围非常有限：它只能表示具有单一输入和单一输出的模型，按顺序逐层进行处理。我们在实践中经常会遇到其他类型的模型，比如多输入模型（如图像及其元数据）、多输出模型（预测数据的不同方面）或具有非线性拓扑结构的模型。
在这种情况下，你可以使用函数式API构建模型。

◆ 模型子类化，也就是将Model类子类化。第3章介绍过如何通过将Layer类子类化来创建自定义层，将Model类子类化的方法与其非常相似：
    •  在__init__()方法中，定义模型将使用的层；
    •  在call()方法中，定义模型的前向传播，重复使用之前创建的层；
    •  将子类实例化，并在数据上调用，从而创建权重。


◆ 函数式模型和子类化模型在本质上有很大区别。函数式模型是一种数据结构——它是由层构成的图，你可以查看、检查和修改它。子类化模型是一段字节码——它是带有call()方法的Python类，其中包含原始代码。这是子类化工作流程具有灵活性的原因——你可以编写任何想要的功能，但它引入了新的限制。
举例来说，由于层与层之间的连接方式隐藏在call()方法中，因此你无法获取这些信息。调用summary()无法显示层的连接方式，利用plot_model()也无法绘制模型拓扑结构。同样，对于子类化模型，你也不能通过访问图的节点来做特征提取，因为根本就没有图。将模型实例化之后，前向传播就完全变成了黑盒子。

◆ 一般来说，函数式API在易用性和灵活性之间实现了很好的平衡。它还可以直接获取层的连接方式，非常适合进行模型可视化或特征提取。如果你能够使用函数式API，也就是说，你的模型可以表示为层的有向无环图，那么我建议使用函数式API而不是模型子类化。


◆ 指标是衡量模型性能的关键，尤其是衡量模型在训练数据上的性能与在测试数据上的性能之间的差异。常用的分类指标和回归指标内置于keras.metrics模块中。大多数情况下，你会使用这些指标。但如果想做一些不寻常的工作，你需要能够编写自定义指标。这很简单！
Keras指标是keras.metrics.Metric类的子类。与层相同的是，指标具有一个存储在TensorFlow变量中的内部状态。与层不同的是，这些变量无法通过反向传播进行更新，所以你必须自己编写状态更新逻辑。这一逻辑由update_state()方法实现。

◆ Keras的回调函数（callback）API可以让model.fit()的调用从纸飞机变为自主飞行的无人机，使其能够观察自身状态并不断采取行动。
回调函数是一个对象（实现了特定方法的类实例），它在调用fit()时被传入模型，并在训练过程中的不同时间点被模型调用。回调函数可以访问关于模型状态与模型性能的所有可用数据，还可以采取以下行动：中断训练、保存模型、加载一组不同的权重或者改变模型状态。回调函数的一些用法示例如下。
    •  模型检查点（model checkpointing）：在训练过程中的不同时间点保存模型的当前状态。
    •  提前终止（early stopping）：如果验证损失不再改善，则中断训练（当然，同时保存在训练过程中的最佳模型）。
    •  在训练过程中动态调节某些参数值：比如调节优化器的学习率。
    •  在训练过程中记录训练指标和验证指标，或者将模型学到的表示可视化（这些表示在不断更新）：fit()进度条实际上就是一个回调函数。

◆ 调用这些方法时，都会用到参数logs。这个参数是一个字典，它包含前一个批量、前一个轮次或前一次训练的信息，比如训练指标和验证指标等。on_epoch_*方法和on_batch_*方法还将轮次索引或批量索引作为第一个参数（整数）。
代码清单7-21给出了一个简单示例，它在训练过程中保存每个批量损失值组成的列表，还在每轮结束时保存这些损失值组成的图。

◆ 利用TensorBoard，你可以做以下工作：
    •  在训练过程中以可视化方式监控指标；
    •  将模型架构可视化；
    •  将激活函数和梯度的直方图可视化；
    •  以三维形式研究嵌入。


◆ 内置的fit()工作流程只针对于监督学习（supervised learning）。监督学习是指，已知与输入数据相关联的目标（也叫标签或注释），将损失计算为这些目标和模型预测值的函数。然而，并非所有机器学习任务都属于这个类别。还有一些机器学习任务没有明确的目标，比如生成式学习（generative learning，第12章将介绍）、自监督学习（self-supervised learning，目标是从输入中得到的）和强化学习（reinforcement learning，学习由偶尔的“奖励”驱动，就像训练狗一样）。

◆ 检索模型权重的梯度时，不应使用tape.gradients(loss, model.weights)，而应使用tape.gradients(loss, model.trainable_weights)。层和模型具有以下两种权重。
    •  可训练权重（trainable weight）：通过反向传播对这些权重进行更新，以便将模型损失最小化。比如，Dense层的核和偏置就是可训练权重。
    •  不可训练权重（non-trainable weight）：在前向传播过程中，这些权重所在的层对它们进行更新。如果你想自定义一层，用于记录该层处理了多少个批量，那么这一信息需要存储在一个不可训练权重中。每处理一个批量，该层将计数器加1。
    在Keras的所有内置层中，唯一具有不可训练权重的层是BatchNormalization层，第9章会介绍它。BatchNormalization层需要使用不可训练权重，以便跟踪关于传入数据的均值和标准差的信息，从而实时进行特征规范化

◆ 这是因为默认情况下，TensorFlow代码是逐行急切执行的，就像NumPy代码或常规Python代码一样。急切执行让调试代码变得更容易，但从性能的角度来看，它远非最佳。
更高效的做法是，将TensorFlow代码编译成计算图，对该计算图进行全局优化，这是逐行解释代码所无法实现的。这样做的语法非常简单：对于需要在执行前进行编译的函数，只需添加@tf.function，

◆ 请记住，调试代码时，最好使用急切执行，不要使用@tf.function装饰器。这样做有利于跟踪错误。一旦代码可以运行，并且你想加快运行速度，就可以将@tf.function装饰器添加到训练步骤和评估步骤中，或者添加到其他对性能至关重要的函数中。


◆ 7.5　本章总结
    •  基于渐进式呈现复杂性的原则，Keras提供了一系列工作流程。你可以顺畅地将它们组合使用。
    •  构建模型有3种方法：序贯模型、函数式API和模型子类化。大多数情况下，可以使用函数式API。
    •  要训练和评估模型，最简单的方式是使用默认方法fit()和evaluate()。
    •  Keras回调函数提供了一种简单方式，可以在调用fit()时监控模型，并根据模型状态自动采取行动。
    •  你也可以通过覆盖train_step()方法来完全控制fit()的效果。
    •  除了fit()，你还可以完全从头开始编写自定义的训练循环。这对研究人员实现全新的训练算法非常有用。

---
##     前六章代码合集

```python
### 1.
 train images.shape(60000,28,28) \\ 6000个矩阵组成的三阶张量，每个矩阵由28*28个整数组成

### 2. 
import tensorflow as tf

## 实现了一个简单的NaiveDense类，创建了两个TensorFlow变量W和b，并定义了一个——call——方法供外部掉用，以实现上述变化

class NaiveDense:  
    def _int_(self, input_size, output_size, activation):
        self.activation = activation

    w_shape = (input_size, output_size)#创建一个形状为（inout_siza,output_size)的矩阵w。并将其随机初始化
    w_initial_value = tf.random(W_shape, minval=0 , maxval = 1e-1)
    self.W = tf.Variable(w_intial_value)

    b_shape = (out_size,) # 创建一个形状为（out_putsize）的零向量b
    b_intial_value = tf.zeros(b_shape)
    self.b = th.Variable(b_intial_value)

    def _call_(self, inputs):# 前向传播
        return self.activation(tf.matual(inputs, self.W) + self.b)

    def weights(self):# 获取该层权重的便捷方法
        return [self.W self.b]

## 创建一个Sequitianl类，封装了层列表，并定义了一个call方法供外部调用

 class NaiveSequential:
    def _init_(self, layers):
        self.layers = layers

    def _call_(self, inputs):
        x = inputs

        for layer in self.layers:
            x = layer(x)
        return x

    def weight(self):
        weights = []
        for layers in self.layers:
            weights += layer.weights
    return weights
# 利用NaiveDense类与NaiveSequittial类创建一个与Keras类似的模型
model = NaiveSequital([
    NaiveDense (input_size = 28*28, outout_size = 512, activateion = tf.nn.relu), 
    NaiveDense(input_size=512, output_size=10,activation = tf.nn.softmax)
])

assert len(modek.weights) == 4#assert语句用于断言模型的权重数量是否为4，如果不是，则会抛出一个异常。这个断言可以用来验证模型的结构是否符合预期。

#批量生成器
import math

class BatchGenerator:
    def _init_(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self. index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batch =math.ceil(len(images) / batch_size)


    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self,batch_size

        return images, labels

#计算梯度
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape : # 运行前向传播，即在GradientTape作用域内计算模型预测值
        predictions = model(images_batch)
        per_sample_losses = tf.keras.loss.sparse_categoraical_crossentropy(labels_batch, prediction)
        average_loss = tf.reduce_mean(per_sample_loss)
    gradients = tape.gradient(average_loss, model.weights) #计算损失相对于权重的梯度，输出gradient是一个列表，每个元素对应model。weights列表中的权重
    updeate_weights(gradient, model.weights)#利用梯度来更新权重
    return average_loss

learning_rate = 1e-3
def updeat_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)


#完整的训练
def fit(model, images, labels, epochs, batch_size = 128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0;
                printf(f"loss at batch {batch_counter} : {loss:2f}")

#评估模型


### 3. 
#利用嵌套的梯度带计算二阶梯度
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradident(position, time)
acceleraion = outer_tape.graident(speed, time)


### 4. 
#Dense层的实现：作为Layers的子类
from tensorflow import keras

class SimpleDense(keras.layers.Layer):#Keras的所有层都继承自Layers基类
    def _init_(self, units, activation=None):
        super()._init_()
        self.units = units
        self.activation = activation

    
    def build(self, inputs_shape):#在build（）中创建权重
        input_dim = input_shape[-1]

        self.W = self.add_weight(shape=input_deim, self.units),tf.Variable(tf.random.uniform(w_shape))
            initializer = "random_normal")
        self.b = self.add_weight(shape=self.units,),
            initializer="zeros")


    def call(self, inputs)：#在call（）方法中定义前向传播计算
        y = tf.matual(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

#一旦训练好了模型，就可以用它来对新的数据进行预测，使用predict（）方法
predictions = model.predict(new_inputs, batch_size = 128)


### 5. 
#影评分类

from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()  #word_index是一个将单词映射为整数索引的字典
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])  #将字典的键和值交换，将整数索引映射为单词
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])  #对评论解码。注意，索引减去了3，因为0、1、2分别是为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）保留的索引

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  #创建一个形状为(len(sequences), dimension)的零矩阵
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.  #将results[i]某些索引对应的值设为1
    return results
x_train = vectorize_sequences(train_data)  #将训练数据向量化
x_test = vectorize_sequences(test_data)  #将测试数据向量化
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32") #标签向量化


from tensorflow import keras
from tensorflow.keras import layers

#模型定义
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
]) 

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])#编译模型


#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


#训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#预测模型
>>> model.predict(x_test)
array([[ 0.98006207]
       [ 0.99758697]
       [ 0.99975556]
       ...,
       [ 0.82167041]
       [ 0.02885115]
       [ 0.65371346]], dtype=float32)


### 6.
#波士顿房价预测

#加载数据集
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = (
    boston_housing.load_data())

#数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#k折交叉验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  #准备验证数据：第k个分区的数据
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(  #准备训练数据：其余所有分区的数据
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()  #构建Keras模型（已编译）
    model.fit(partial_train_data, partial_train_targets,  #训练模型（静默模式，verbose=0）
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)  #在验证数据上评估模型
    all_scores.append(val_mae)


#保存没折的验证分数
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  #准备验证数据：第k个分区的数据
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(  #准备训练数据：其余所有分区的数据
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()  #构建Keras模型（已编译）
    history = model.fit(partial_train_data, partial_train_targets,  #训练模型（静默模式，verbose=0）
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)


#绘制验证MAE曲线
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

#训练最终模型
model = build_model()  #一个全新的已编译模型
model.fit(train_data, train_targets,  #在所有训练数据上训练模型
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

#预测
>>> predictions = model.predict(test_data)
>>> predictions[0]
array([9.990133], dtype=float32)


```
----
## 序贯模型（Sequitianl类）

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax" )
])

>>> model.build(input_shape = (None, 3))#检索模型权重
>>> model.weights #调用模型的build方法，模型样本形状应该是（3，）。输入形状可以是任意大小

>>>model.summary()#模型搭建完可以通过summary方法显示模型内容
Model: "sequential_1"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
dense_2 (Dense)             (None, 64)                256
_________________________________________________________________
dense_3 (Dense)             (None, 10)                650
=================================================================
Total params: 906
Trainable params: 906
Non-trainable params: 0
_________________________________________________________________



>>> model = keras.Sequential(name="my_example_model")
>>> model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
>>> model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
>>> model.build((None, 3))
>>> model.summary()
Model: "my_example_model"
_________________________________________________________________
Layer (type)                 Output Shape             Param #
=================================================================
my_first_layer (Dense)       (None, 64)               256
_________________________________________________________________
my_last_layer (Dense)        (None, 10)               650
=================================================================
Total params: 906
Trainable params: 906
Non-trainable params: 0
_________________________________________________________________

```
___

## 函数式API

```python
#通过给定输入和目标组成的列表来训练模型
import numpy as np

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))  #本行及以下2行)虚构的输入数据
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))  #(本行及以下1行)虚构的目标数据
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data])


```
## 模型子类化
在_init_()方法中，定义模型将使用的层；
在_call()方法中，定义模型的前向传播，重复使用之前创建的层；
将子类实例化，并在数据上调用，从而创建权重
```python
#使用Model子类重新实现客户支持工单管理模型

class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        super().__init__()  #不要忘记调用super()构造函数！
        self.concat_layer = layers.Concatenate()  #(本行及以下3行)在构造函数中定义子层
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax")

    def call(self, inputs):  #在call()方法中定义前向传播
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],  #(本行及以下1行)参数loss和metrics的结构必须与call()返回的内容完全匹配——这里是两个元素组成的列表
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit({"title": title_data,  #(本行及以下2行)输入数据的结构必须与call()方法的输入完全匹配——这里是一个字典，字典的键是title、text_body和tags
           "text_body": text_body_data,
           "tags": tags_data},
          [priority_data, department_data],  #目标数据的结构必须与call()方法返回的内容完全匹配——这里是两个元素组成的列表
          epochs=1)
model.evaluate({"title": title_data,
                "text_body": text_body_data,
                "tags": tags_data},
               [priority_data, department_data])
priority_preds, department_preds = model.predict({"title": title_data, "text_body": text_body_data, "tags": tags_data})



from tensorflow.keras.datasets import mnist

def get_mnist_model():  #创建模型（我们将其包装为一个单独的函数，以便后续复用）
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

(images, labels), (test_images, test_labels) = mnist.load_data()  #加载数据，保留一部分数据用于验证
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(optimizer="rmsprop",  #(本行及以下2行)编译模型，指定模型的优化器、需要最小化的损失函数和需要监控的指标
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels,  #(本行及以下2行)使用fit()训练模型，可以选择提供验证数据来监控模型在前所未见的数据上的性能
          epochs=3,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)  #使用evaluate()计算模型在新数据上的损失和指标
predictions = model.predict(test_images)  #使用predict()计算模型在新数据上的分类概率


#在fit方法中使用callbacks参数

callbacks_list = #通过fit()的callbacks参数将回调函数传入模型中，该参数接收一个回调函数列表，可以传入任意数量的回调函数
    keras.callbacks.EarlyStopping(  #如果不再改善，则中断训练
        monitor="val_accuracy",  #监控模型的验证精度
        patience=2,  #如果精度在两轮内都不再改善，则中断训练
    ),
    keras.callbacks.ModelCheckpoint(  #在每轮过后保存当前权重
        filepath="checkpoint_path.keras",  #模型文件的保存路径
        monitor="val_loss",  # (本行及以下1行)这两个参数的含义是，只有当val_loss改善时，才会覆盖模型文件，这样就可以一直保存训练过程中的最佳模型
        save_best_only=True,
    )
]
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])  #监控精度，它应该是模型指标的一部分
model.fit(train_images, train_labels,  # (本行及以下3行)因为回调函数要监控验证损失和验证指标，所以在调用fit()时需要传入validation_data（验证数据）
          epochs=10,
          callbacks=callbacks_list,
          validation_data=(val_images, val_labels))


#编写自定义调回函数

on_epoch_begin(epoch, logs)  #在每轮开始时被调用
on_epoch_end(epoch, logs)  #在每轮结束时被调用
on_batch_begin(batch, logs)  #在处理每个批量之前被调用
on_batch_end(batch, logs)  #在处理每个批量之后被调用
on_train_begin(logs)  #在训练开始时被调用
on_train_end(logs)  #在训练结束时被调用

#通过对Callback类子类化来创建自定义回调函数

from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
                 label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []


```

