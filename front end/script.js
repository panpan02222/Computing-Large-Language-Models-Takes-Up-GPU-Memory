const calculateButton = document.getElementById('calculate-button');
const dataTypeSelect = document.getElementById('dataType');
const paramsInput = document.getElementById('params');
const optimizerSelect = document.getElementById('optimizer');
const sequenceLengthInput = document.getElementById('sequence-length');
const batchSizeInput = document.getElementById('batchsize');
const hiddenSizeInput = document.getElementById('hidden-size');
const attentionHeadsInput = document.getElementById('attention-heads');
const transformerLayersInput = document.getElementById('transformer-layers');

// 并行方式计算
// 获取选中的复选框的值并存储在常量中
// 获取选中的复选框的值并存储在常量中
function getSelectedValues() {
  const selectedValues = {
    tp_num: null,
    pp_num: null,
    sp_selected: false,
    cjs_selected: false
  };

  if (document.getElementById('tp').checked) {
    console.log("Tensor 并行选中");
    selectedValues.tp_num = parseInt(document.getElementById('tp-num').value);
  }

  if (document.getElementById('sp').checked) {
    console.log("Sequence 并行选中");
    selectedValues.sp_selected = true;
  }

  if (document.getElementById('pp').checked) {
    console.log("Pipeline 并行选中");
    selectedValues.pp_num = parseInt(document.getElementById('pp-num').value);
  }

  if (document.getElementById('chongjisuan').checked) {
    console.log("重计算选中");
    selectedValues.cjs_selected = true;
  }

  return selectedValues;
}


const memoryValue = document.getElementById('memory-value');

calculateButton.addEventListener('click', function() {
  // 获取选中的训练方法及其参数
  const selectedValues = getSelectedValues();
  const { tp_num, pp_num, sp_selected, cjs_selected } = selectedValues;

  console.log(tp_num, pp_num, sp_selected, cjs_selected);

  const dataType = dataTypeSelect.value;
  const paramsInBillions = parseFloat(paramsInput.value);
  const optimizer = optimizerSelect.value;
  const sequenceLength = parseInt(sequenceLengthInput.value);
  const batchSize = parseInt(batchSizeInput.value);
  const hiddenSize = parseInt(hiddenSizeInput.value);
  const attentionHeads = parseInt(attentionHeadsInput.value);
  const transformerLayers = parseInt(transformerLayersInput.value);

  if (isNaN(paramsInBillions)) {
    alert('请输入有效数字.');
    return;
  }

  // 模型参数量
  const params = paramsInBillions * 1e9; // 将十亿转换为参数数量
  // 模型精度
  const typesize = getDataTypeSize(dataType);

  const modelMemory = params * typesize; // 模型参数量乘以数据类型大小
  // 优化器状态值
  const optimizerState = getOptimizerState(optimizer, params);
  // 梯度值
  const gradientValue = getGradientValue(typesize, params);
  // 激活值
  const activationValue = getActivateValue(sequenceLength, batchSize, hiddenSize, attentionHeads, transformerLayers, selectedValues);

  // 计算内存使用量
  const memoryUsage = calculateMemory(modelMemory, gradientValue, optimizerState, activationValue, tp_num, sp_selected, pp_num);

  // 使用格式化更新内存值显示
  memoryValue.textContent = formatMemory(memoryUsage);
});

// 修改后的 getActivateValue 函数定义
function getActivateValue(s, b, h, a, L, trainingMethods) {
  let activationValue = 0;
  let lambda = 1 / (1024 * 1024 * 1024);

  if (trainingMethods.cjs_selected) {
    if (trainingMethods.tp_num && trainingMethods.sp_selected) {
      console.log("选择重计算 + Tensor并行 + 序列并行");
      activationValue = s * b * h * (34 / trainingMethods.tp_num) * L * lambda;
    } else if (trainingMethods.tp_num) {
      console.log("选择重计算 + Tensor并行");
      activationValue = s * b * h * (10 + 24 / trainingMethods.tp_num) * L * lambda;
    } else {
      console.log("全部重计算");
      activationValue = s * b * h * 2 * L * lambda;
    }
  } else {
    if (trainingMethods.tp_num && trainingMethods.sp_selected) {
      console.log("Tensor并行 + 序列并行");
      activationValue = s * b * h * (34 + 5 * a * s / h) / trainingMethods.tp_num * L * lambda;
    } else if (trainingMethods.tp_num) {
      console.log("Tensor并行（基线）");
      activationValue = s * b * h * (10 + 24 / trainingMethods.tp_num + 5 * a * s / h) * L * lambda;
    } else {
      console.log("无重计算无并行");
      activationValue = s * b * h * (34 + 5 * a * s / h) * L * lambda;
    }
  }

  return activationValue;
}


function getDataTypeSize(dataType) {
  switch (dataType) {
    case 'FP32':
      return 4;
    case 'FP16':
      return 2;
    case 'BF16':
      return 2;
    case 'INT16':
      return 2;
    case 'INT8':
      return 1;
    default:
      throw new Error('Invalid data type: ' + dataType);
  }
}

function getOptimizerState(optimizer, params) {
  let additionalMemoryPerParam = 0;
  let modelMemory = 4;
  let momentumMemory = 4;
  let varianceMemory = 4;

  if (optimizer === 'AdamW') {
    additionalMemoryPerParam = (modelMemory + momentumMemory + varianceMemory / 4) * params; // AdamW uses 3 times the parameter size
  } else if (optimizer === 'Adam') {
    additionalMemoryPerParam = (modelMemory + momentumMemory + varianceMemory) * params; // Adam uses 3 times the parameter size
  } else if (optimizer === '8-bit-opti') {
    additionalMemoryPerParam = (modelMemory + (momentumMemory / 4) + (varianceMemory / 4)) * params; // 八位优化器 uses 2 times the parameter size (adjusted)
  } else if (optimizer === '4-bit-opti') {
    additionalMemoryPerParam = ((modelMemory / 2) + (momentumMemory / 4) + (varianceMemory / 4)) * params; // 四位优化器 uses 2 times the parameter size (adjusted)
  }

  return additionalMemoryPerParam;
}



function getGradientValue(typesize, params) {
  let gradientValue = 0;
  gradientValue = typesize * params;
  return gradientValue;
}
// 激活值计算
// function getActivateValue(s, b, h, a, L, trainingMethods) {
//   let activationValue = 0;
//   let lambda = 1 / (1024 * 1024 * 1024);

//   if (trainingMethods.cjs_selected) {
//     if (trainingMethods.tp_num && trainingMethods.sp_selected) {
//       console.log("选择重计算 + Tensor并行 + 序列并行");
//       activationValue = s * b * h * (34 / trainingMethods.tp_num) * L * lambda;
//     } else if (trainingMethods.tp_num) {
//       console.log("选择重计算 + Tensor并行");
//       activationValue = s * b * h * (10 + 24 / trainingMethods.tp_num) * L * lambda;
//     } else {
//       console.log("全部重计算");
//       activationValue = s * b * h * 2 * L * lambda;
//     }
//   } else {
//     if (trainingMethods.tp_num && trainingMethods.sp_selected) {
//       console.log("Tensor并行 + 序列并行");
//       activationValue = s * b * h * (34 + 5 * a * s / h) / trainingMethods.tp_num * L * lambda;
//     } else if (trainingMethods.tp_num) {
//       console.log("Tensor并行（基线）");
//       activationValue = s * b * h * (10 + 24 / trainingMethods.tp_num + 5 * a * s / h) * L * lambda;
//     } else {
//       console.log("无重计算无并行");
//       activationValue = s * b * h * (34 + 5 * a * s / h) * L * lambda;
//     }
//   }

//   return activationValue;
// }



// // 无重计算无并行
// function getActivateValue(s,b,h,a,L) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(34 + 5 *a * s/h) * L * lambda;
//   return activationValue;
// }

// // Tensor并行 + 序列并行
// function getActivateValue(s,b,h,a,L,t) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(34 + 5 *a * s/h)/t * L * lambda;
//   return activationValue;
// }

// // Tensor并行（基线）
// function getActivateValue_0(s,b,h,a,L,t) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(10 + 24/t + 5 * a * s/h) * L * lambda;
//   return activationValue;
// }

// // 选择重计算 + Tensor并行
// function getActivateValue_0(s,b,h,a,L,t) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(10 + 24/t) * L * lambda;
//   return activationValue;
// }

// // 选择冲计算 + Tensor并行 + 序列并行
// function getActivateValue(s,b,h,a,L,t) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(34/t) * L * lambda;
//   return activationValue;
// }

// // 全部重计算
// function getActivateValue(s,b,h,L) {
//   let activationValue = 0;
//   let lambda = 1/(1024*1024*1024);
//   activationValue = s * b * h *(2) * L * lambda;
//   return activationValue;
// }


function calculateMemory(modelMemory, optimizerState, gradientValue, activationValue, tp_num, sp_num, pp_num) {
  const modelMemoryGB = modelMemory / (1024 * 1024 * 1024); // 转换为GB
  const optimizerStateGB = optimizerState / (1024 * 1024 * 1024); // 转换为GB
  const gradientValueGB = gradientValue / (1024 * 1024 * 1024); // 转换为GB

  console.log("modelMemoryGB", modelMemoryGB, "GB")
  console.log("optimizerStateGB", optimizerStateGB, "GB")
  console.log("gradientValueGB", gradientValueGB, "GB")
  console.log("activationValue", activationValue, "GB")

  console.log("tp_num", tp_num)
  console.log("sp_num", sp_num)
  console.log("pp_num", pp_num)
  const memoryUsage = modelMemoryGB/(pp_num*tp_num) + optimizerStateGB + gradientValueGB/pp_num + activationValue/tp_num;

  return memoryUsage;
}

function formatMemory(memoryUsage) {
  return memoryUsage.toFixed(2) + ' GB';
}
