import * as tf from '@tensorflow/tfjs';
import { tokenize, removeStopWords, createDictionaryForTotalTokenCountInAllDocs } from "./utils";
import { getInverseDocumentFrequency, getTfIdf, getTermFrequency } from './embedding-utils/tfidf'
import { trainValTestSplit } from './ml-utils'
import * as tfvis from '@tensorflow/tfjs-vis';
// const csvUrl = 'data/toxic_data_sample.csv'

// const csvUrl = 'data/toxicdatasample.csv';
// const csvUrl = 'http://localhost:1234/toxicdatasample.csv';
const csvUrl = 'http://localhost:1234/toxicdatamid.csv';
const EMBEDDING_SIZE = 8000;
const TRAINING_EPOCH = 15;
const render = true;
const history = [];
const surface = { name: 'onEpochEnd Performance', tab: 'Training' }
const batchHistory = [];
const batchSurface = { name: 'onBatchEnd Performance', tab: 'Training' };
const MODEL_ID = 'toxicity_detector_tfidf';
const IDF_STORAGE_ID = 'toxicity-idfs';
const DICTIONARY_STORAGE_ID = 'toxicity-tfidf-dictionary';

const readRawData = () => {
    // load data
    console.log('readRawData called');
    console.log('csvUrl', csvUrl);

    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    });
    return readData;
}

const plotBarGraph = (labels) => {
    labelOutput = labels.reduce((acc, label) => {
        acc[label] = acc[label] === undefined ? 1 : acc[label] += 1;
        return acc;
    },
        {});
    // console.log(labelOutput);
    const labelOutputArray = Object.keys(labelOutput).map((val) => ({ 'index': val, 'value': labelOutput[val] }));
    // console.log('labelOutputArray: ', labelOutputArray)
    tfvis.render.barchart({ name: 'Exploration', tab: 'Charts' }, labelOutputArray)
}

eachTokenCountDictionary = {};
comments = [];
const documentTokens = [];

const run = async () => {
    const labels = [];
    console.log('run called');
    const data = readRawData();
    await data.forEachAsync((item) => {
        const comment = item['xs']['comment_text'];
        const trimmedComment = comment.toLowerCase().trim();
        comments.push(trimmedComment);
        // const tokens = removeStopWords(tokenize(trimmedComment));
        const tokens = tokenize(trimmedComment);
        // console.log('tokens length: ', tokens.length);
        documentTokens.push(tokens);
        // console.log('documentTokens length: ', documentTokens.length);
        createDictionaryForTotalTokenCountInAllDocs(tokens, eachTokenCountDictionary);
        labels.push(item['ys']['toxic'])
    });
    // console.log(labels);
    if (render) {
        plotBarGraph(labels);
    }

    // console.log('eachTokenCountDictionary: ', eachTokenCountDictionary)
    // console.log('eachTokenCountDictionary Length: ', Object.keys(eachTokenCountDictionary).length)

    const sortDictionary = sortDictionaryByValue(eachTokenCountDictionary);
    if (EMBEDDING_SIZE > sortDictionary.length) {
        EMBEDDING_SIZE = sortDictionary.length;
    }

    const dictionary = sortDictionary.slice(0, EMBEDDING_SIZE).map(row => row[0]);
    console.log('dictionary: ', dictionary);
    // console.log('document Tokens: ', documentTokens)
    const idfs = getInverseDocumentFrequency(documentTokens, dictionary);
    console.log(idfs);
    const ds = prepareData(dictionary, idfs);
    // ds.forEachAsync(item => console.log(item))

    // const ds = prepareDataUsingGenerator(comments, labels, dictionary, idfs);
    // console.log('ds:', (await ds.toArray()).length);
    // ds.forEachAsync(item => console.log(item))
    // console.log('documentTokens: ', documentTokens.length);
    const SEED = 7687547
    const BATCH_SIZE = 16;
    const { trainingDataset, validationDataset, testDataset } = trainValTestSplit(ds, documentTokens.length, SEED, BATCH_SIZE);
    // trainingDataset.forEachAsync(item => console.log(item));
    let model = buildModel();
    if (render) {

        tfvis.show.modelSummary({
            name: 'Model Summary',
            tab: 'Model'
        },
            model
        )

    }

    model = await trainModel(model, trainingDataset, validationDataset);
    await evaluateModel(model, testDataset);
    await getMoreEvaluationSummaries(model, testDataset);
    const exportResult = await exportModel(model, MODEL_ID, IDF_STORAGE_ID, DICTIONARY_STORAGE_ID, idfs, dictionary);
    // const loadedModel = await loadModel();
    // loadedModel.summary();

    // const sentence1 = 'It is great';
    // const predictedClass1 = predictModel(sentence1, model);
    // console.log('predicted class is: ', predictedClass1)

    // const sentence2 = 'What the fuck';
    // const predictedClass2 = predictModel(sentence2, model);
    // console.log('predicted class is: ', predictedClass2)
}

const prepareDataUsingGenerator = (comments, dictionary, labels, idfs) => {

    function* getFeatures() {
        for (let i = 0; i < comments.length; i++) {
            const encoded = encoder(comments[i], dictionary, idfs)
            yield tf.tensor2d([encoded], [1, dictionary.length])
        }
    }

    function* getLabels() {
        for (let i = 0; i < labels.length; i++) {
            yield tf.tensor2d([labels[i]], [1, 1])
        }
    }

    const xs = tf.data.generator(getFeatures);
    const ys = tf.data.generator(getLabels);

    return tf.data.zip({ xs, ys })
}

const sortDictionaryByValue = (dict) => {
    const items = Object.keys(dict).map((key) => ([key, dict[key]]));
    items.sort((a, b) => b[1] - a[1]);
    // console.log(items);
    return items;
}

const encoder = (sentence, dictionary, idfs) => {
    const tokens = removeStopWords(tokenize(sentence));
    const tfs = getTermFrequency(tokens, dictionary);
    // console.log('tfs: ', tfs);
    const tfIdfs = getTfIdf(tfs, idfs);
    return tfIdfs;
}

const prepareData = (dictionary, idfs) => {

    const preProcess = ({ xs, ys }) => {
        const comment = xs['comment_text'];
        const trimmedComment = comment.toLowerCase().trim();

        const encoded = encoder(trimmedComment, dictionary, idfs);
        return {
            xs: tf.tensor2d([encoded], [1, dictionary.length]),
            ys: tf.tensor2d([ys['toxic']], [1, 1])
        }
    }

    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    })
        .map(preProcess);
    return readData;
}

const messageCallback = new tf.CustomCallback({
    onEpochEnd: async (epoch, logs) => {

        history.push(logs)
        // console.log('Logs: ' + JSON.stringify(logs))
        // console.log('Epoch: ' + epoch + ' loss: ' + logs.loss)
        if (render) {
            tfvis.show.history(surface, history, ['loss', 'val_loss', 'acc', 'val_acc'])
        }
    },
    onBatchEnd: async (batch, logs) => {
        batchHistory.push(logs);
        // console.log('batch: ', JSON.stringify(batch))
        if (render) {
            tfvis.show.history(batchSurface, batchHistory, ['loss', 'val_loss', 'acc', 'val_acc'])
        }
    }
})

const earlyStoppingCallback = tf.callbacks.earlyStopping({
    monitor: 'acc',
    minDelta: 0.3,
    patience: 5,
    verbose: 1
})

const buildModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 5, inputShape: [EMBEDDING_SIZE], activation: 'relu' }))
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))
    model.compile({ loss: 'binaryCrossentropy', optimizer: tf.train.adam(0.06), metrics: ['accuracy'] })
    model.summary();
    return model;
}

const trainModel = async (model, trainingDataset, validationDataset) => {
    await model.fitDataset(trainingDataset, {
        epochs: TRAINING_EPOCH,
        validationDataset: validationDataset,
        // callbacks: [messageCallback, earlyStoppingCallback]
        callbacks: [messageCallback]
    });
    return model;
}

const evaluateModel = async (model, testDataset) => {
    modelResults = await model.evaluateDataset(testDataset);
    // console.log('modelResults: ', modelResults);
    // console.log('modelResults[0].dataSync()', modelResults[0].dataSync());
    const testLoss = modelResults[0].dataSync()[0];
    const testAcc = modelResults[1].dataSync()[0];
    console.log(`Loss on Test Dataset : ${testLoss.toFixed(4)}`);
    console.log(`Accuracy on Test Dataset : ${testAcc.toFixed(4)}`);
}

const getMoreEvaluationSummaries = async (model, testDataset) => {
    allActualLabels = [];
    allPredictedLabels = [];

    await testDataset.forEachAsync(item => {
        const accLabels = item['ys'].dataSync();
        accLabels.forEach(label => allActualLabels.push(label));

        // const features = item['xs'].dataSync();
        // console.log(features);

        const features = item['xs'];
        // console.log(features);
        // console.log(tf.squeeze(features, 1));
        const predict = model.predictOnBatch(tf.squeeze(features, 1));
        const predictions = tf.round(predict).dataSync();
        predictions.forEach(prediction => allPredictedLabels.push(prediction));

        // console.log(predict);
        // predictions.forEachAsync((prediction) => allPredictedLabels.push(prediction));

    });
    // console.log(allActualLabels)
    //console.log(allPredictedLabels);

    // create actual and predicted label tensors
    const allActualLablesTensor = tf.tensor1d(allActualLabels);
    const allPredictedLablesTensor = tf.tensor1d(allPredictedLabels);

    // calculate accuracy result
    const accuracyResult = await tfvis.metrics.accuracy(allActualLablesTensor, allPredictedLablesTensor);
    // console.log('allActualLabels.length: ' + allActualLabels + ' allPredictedLabels.lemgth: ' + allPredictedLabels)
    console.log(`Accuracy result : ${accuracyResult}`);

    // // calculate per class accuracy result
    const perClassAccuracyResult = await tfvis.metrics.perClassAccuracy(allActualLablesTensor, allPredictedLablesTensor);
    console.log(`Per Class Accuracy result : ${JSON.stringify(perClassAccuracyResult, null, 2)}`);

    // // create confusion matrix report
    const confusionMatrixResult = await tfvis.metrics.confusionMatrix(allActualLablesTensor, allPredictedLablesTensor);
    const confusionMatrixVizResult = { "values": confusionMatrixResult };
    console.log(`confusion matrix : \n ${JSON.stringify(confusionMatrixVizResult, null, 2)}`);
    const surface = { tab: 'Evaluation', name: 'Confusion Matrix' };
    if (render) {
        tfvis.render.confusionMatrix(surface, confusionMatrixVizResult);
    }

}

const exportModel = async (model, modelId, idfStorageId, dictStorageId, idfs, dictionary) => {
    const modelPath = `localstorage://${modelId}`;
    const saveModelResults = await model.save(modelPath);
    console.log('model exported', saveModelResults);

    localStorage.setItem(idfStorageId, JSON.stringify(idfs));
    localStorage.setItem(dictStorageId, JSON.stringify(dictionary));
    console.log('dictionary and IDFs exported');

    return exportModel;
}

const loadModel = async () => {
    const modelPath = `localstorage://${MODEL_ID}`;
    const models = await tf.io.listModels();
    if (models[modelPath]) {
        const model = await tf.loadLayersModel(modelPath);
        return model
    }
    else {
        console.log('no model available');
        return null;
    }

}

export const load = () => {
    return loadModel();
}

const predictModel = (sentence, model) => {
    const idfObject = localStorage.getItem(IDF_STORAGE_ID);
    const idfs = JSON.parse(idfObject);

    const dictionaryObject = localStorage.getItem(DICTIONARY_STORAGE_ID);
    const dictionary = JSON.parse(dictionaryObject);

    const encoded = encoder(sentence.toLowerCase().trim(), dictionary, idfs);
    const encodedTensor = tf.tensor2d([encoded], [1, dictionary.length]);

    const predictionResult = model.predict(encodedTensor);

    const predictionScore = predictionResult.dataSync();
    const predictedClass = (predictionScore < 0.5) ? 'not-toxic' : 'toxic';
    const result = `Probability: ${predictionScore} and Class: ${predictedClass}`;
    console.log(result);
    
    return predictedClass;
}

export const predict = (sentence, model) => {
    console.log("predict using custom model");
    return predictModel(sentence, model);
};

export const  train = async () => {
    tf.tidy(async () => {
        await run();
    })

}