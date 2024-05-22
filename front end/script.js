const calculateButton = document.getElementById('calculate-button');
const dataTypeSelect = document.getElementById('dataType');
const paramsInput = document.getElementById('params');
const memoryValue = document.getElementById('memory-value');

calculateButton.addEventListener('click', function() {
  const dataType = dataTypeSelect.value;
  const paramsInBillions = parseFloat(paramsInput.value);

  if (isNaN(paramsInBillions)) {
    alert('Please enter a valid number for the number of parameters (billions).');
    return;
  }

  const params = paramsInBillions * 1e9; // Convert billions to parameters
  const typesize = getDataTypeSize(dataType);
  const memoryUsage = calculateMemory(typesize, params);

  // Update memory value display with formatting
  memoryValue.textContent = formatMemory(memoryUsage);
});

function getDataTypeSize(dataType) {
  switch (dataType) {
    case 'FP32':
      return 4;
    case 'FP16':
      return 2;
    case 'BF16':
      return 2;
    case 'INT16':
        return 2
    case 'INT8':
      return 1;
    default:
      throw new Error('Invalid data type: ' + dataType);
  }
}

function calculateMemory(typesize, params) {
  return params * typesize;
}

function formatMemory(memoryUsage) {
  if (memoryUsage < 1e9) {
    return memoryUsage.toFixed(2) + ' bytes';
  } else if (memoryUsage < 1e12) {
    return (memoryUsage / 1e9).toFixed(2) + ' GB';
  } else {
    return (memoryUsage / 1e12).toFixed(2) + ' TB';
  }
}
