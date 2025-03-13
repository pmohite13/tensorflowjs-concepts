const trainValTestSplit = (dataset, nrows, seed, batchsize, splitPercent = 70) => {
    console.log('nrows: ', nrows);

    const trainingValidationSplitPercent = splitPercent - 10;

    const trainingCount = Math.round(nrows * (trainingValidationSplitPercent/100));
    const traingValidationCount = Math.round(nrows * (splitPercent / 100));

    console.log('trainingCount: ', trainingCount);    
    console.log('traingValidationCount: ', traingValidationCount);

    const trainingValidationData = dataset.shuffle(nrows, seed).take(traingValidationCount);

    const testDataset = dataset.shuffle(nrows, seed).skip(traingValidationCount).batch(batchsize);

    const trainingDataset = trainingValidationData.take(trainingCount).batch(batchsize);

    const validationDataset = trainingValidationData.skip(trainingCount).batch(batchsize);

    return { trainingDataset, validationDataset, testDataset }
}

export { trainValTestSplit }