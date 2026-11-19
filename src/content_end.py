"""报告正文：第 4 章总结与展望、第 5 章参考文献、第 6 章课程收获。"""

BLOCKS = []
A = BLOCKS.append

A(("h1", "4　总结与展望"))

A(("h2", "4.1　工作总结"))
A(("p", "本文围绕示功图的自动分类问题，完成了一套从数据到部署的完整流程。主要工作包括四个方面："
        "一是梳理了采油工程与示功图诊断的背景知识，调研了从几何特征、专家系统、传统机器学习到深度学习的四类方法及其优缺点；"
        "二是针对本数据集类别不均衡、图形干净规整的特点，设计了轻量卷积网络 SGTNet-S，并结合随机仿射增强、类别加权交叉熵、"
        "标签平滑与余弦退火完成训练，在 CPU 上 15 分钟即可收敛；三是搭建了像素+PCA+SVM、HOG+SVM、形状特征+SVM 与 ResNet18 四类对照组，"
        "从精度、规模、速度、鲁棒性四个维度进行了系统比较；四是通过消融实验、鲁棒性测试、Grad-CAM 热力图与错分样本分析，"
        "解释了模型的行为与失败模式。"))
A(("p", "实验得到的主要结论是：在本数据集上，各类方法的精度都已接近饱和（99.5% 以上），模型容量不再是瓶颈；"
        "轻量 CNN 的精度与最优传统方法相当，但在噪声、模糊、旋转与遮挡条件下的稳定性明显更好，且推理速度满足井口实时诊断的需求；"
        "Grad-CAM 显示模型关注的是台阶、弧顶、锯齿等关键形态区域，与人工诊断的经验一致。这些结果对理解“深度学习在示功图诊断中究竟带来了什么”"
        "具有参考价值：它带来的不是更高的实验室精度，而是更强的鲁棒性、更低的特征工程成本与更好的可迁移性。"))

A(("h2", "4.2　不足与展望"))
A(("p", "本文的不足主要有三点。第一，数据来源单一。数据集经过清洗与统一处理，背景纯白、笔画均匀，与真实井场采集的示功图（含网格、坐标轴、"
        "噪声、残缺）差异较大，模型在现场的泛化能力有待检验。第二，类别缺少语义标签。十类工况只有编号，无法与“供液不足”“气锁”“泵漏失”等"
        "工程术语对应，也就无法从物理机理上解释某两类为何容易混淆。第三，鲁棒性实验中的扰动是人工合成的，与真实的传感器漂移、"
        "雨雪干扰、通信丢包仍有差距。"))
A(("p", "后续工作可以从三个方向展开：一是引入真实井场数据，研究跨区块、跨泵型的迁移学习与领域自适应，解决“实验室精度高、现场效果差”的普遍问题；"
        "二是结合载荷—位移原始曲线与时序信息，用一维卷积或时序模型替代图像输入，避免二值化与描边造成的信息损失；"
        "三是探索轻量化部署与在线学习，把模型压缩后部署到井口边缘设备，并利用新产生的数据持续更新，形成“诊断—反馈—再训练”的闭环。"
        "此外，把示功图与电流、动液面、产量等多源数据融合，也是提高复杂工况识别率的可行途径。"))

A(("h1", "5　参考文献"))
A(("ref", "[1] Koroteev D, Tekic Z. Artificial intelligence in oil and gas upstream: trends, challenges, and scenarios for the future[J]. Energy and AI, 2021, 3: 100041."))
A(("ref", "[2] 张琪. 采油工程原理与设计[M]. 东营: 中国石油大学出版社, 2000."))
A(("ref", "[3] 万仁溥. 采油工程手册[M]. 北京: 石油工业出版社, 2000."))
A(("ref", "[4] 周斌, 王延江, 刘伟锋, 等. 基于Hessian正则化支持向量机的多视角协同识别抽油机井工况方法[J]. 石油学报, 2018, 39(12): 1429-1436."))
A(("ref", "[5] 李汉周, 段志刚, 朱苏青, 等. 基于深度主动学习的示功图诊断方法及应用[J]. 石油化工自动化, 2023, 59(6)."))
A(("ref", "[6] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324."))
A(("ref", "[7] Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[C]//Proceedings of the 32nd International Conference on Machine Learning. Lille: PMLR, 2015: 448-456."))
A(("ref", "[8] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas: IEEE, 2016: 2818-2826."))
A(("ref", "[9] Kingma D P, Ba J. Adam: a method for stochastic optimization[C]//International Conference on Learning Representations. San Diego, 2015."))
A(("ref", "[10] Loshchilov I, Hutter F. Decoupled weight decay regularization[C]//International Conference on Learning Representations. New Orleans, 2019."))
A(("ref", "[11] Dalal N, Triggs B. Histograms of oriented gradients for human detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. San Diego: IEEE, 2005: 886-893."))
A(("ref", "[12] Cortes C, Vapnik V. Support-vector networks[J]. Machine Learning, 1995, 20(3): 273-297."))
A(("ref", "[13] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas: IEEE, 2016: 770-778."))
A(("ref", "[14] Selvaraju R R, Cogswell M, Das A, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 618-626."))

A(("h1", "6　课程收获"))
A(("p", "一学期的《石油与人工智能》课程，让我第一次比较系统地意识到：石油工程正在从一个“经验驱动”的行业，转向“数据与模型驱动”的行业。"
        "过去在专业课上学到的示功图、动液面、泵效，是工程师拿着纸笔在井场上画的、算的；而现在，同样一张示功图可以变成 96×96 的像素矩阵，"
        "被卷积网络在毫秒级判断出工况。这种变化不是把专业知识“替换掉”，而是给专业知识换了一种表达方式：懂原理的人知道该看图形的哪一部分，"
        "懂算法的人知道怎么让机器也去看那一部分，两者结合才能真正落地。"))
A(("p", "在完成结课项目的过程中，我体会最深的是“数据比模型更重要”。我原以为需要很深的网络才能取得好效果，实验结果却是："
        "一个 3 秒训练完的 PCA+SVM 就能达到 99.76%，而 11 M 参数的 ResNet18 也只多了 0.01 个百分点。真正拉开差距的，是加入噪声、旋转、"
        "遮挡之后各模型的表现——这让我理解了为什么工业界反复强调数据质量、标注一致性与场景覆盖，而不是一味追求更大的模型。"
        "同时，用 Grad-CAM 看到网络把注意力放在台阶和弧顶上时，我第一次直观地感受到“可解释性”对工程应用的意义："
        "现场工程师不会轻易相信一个说不出理由的结论。"))
A(("p", "另一个收获是对“人工智能 + 传统能源”这件事的认识发生了变化。以前觉得 AI 进入石油行业是“降维打击”，现在更愿意把它看作一种工具："
        "它能处理人看不过来的海量数据、能不知疲倦地巡检、能把老师傅的经验固化成模型，但它替代不了对油藏、对工艺、对现场的理解。"
        "示功图分类只是采油智能化的一个小环节，背后还有工况诊断、参数优化、产量预测、设备健康管理一整套体系。"
        "越是深入，越觉得需要补的课多：既要把采油工程的基本概念学扎实，也要能读懂一篇深度学习论文、能写出一段靠谱的代码。"))
A(("p", "最后，感谢课程中老师对油田智能化应用的讲解，让我看到了课堂知识与生产现场之间的距离，也看到了自己未来可以努力的方向。"
        "如果以后有机会进入油田工作，我希望自己能成为那个“既懂泵、又懂模型”的人。"))
