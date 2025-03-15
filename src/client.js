import * as tf from "@tensorflow/tfjs";
import * as tfvis from '@tensorflow/tfjs-vis';
// import "@tensorflow/tfjs-backend-wasm";
import * as model from './model';

import $ from "jquery";
import "materialize-css";
import "material-icons";
import "./main.scss";

M.AutoInit();

console.log(tf.version);

console.log(tf.getBackend())// tf.setBackend('wasm');

// tf.ready().then(() => {
//     console.log(tf.getBackend())
// }) 

const init = async () => {
    await tf.ready();

    const init_message = "Powered by TensorFlow.js - version: " + tf.version.tfjs + " with backend : " + tf.getBackend();
    console.log('$', $);
    $('#init').text(init_message);

    // console.log(tf.getBackend());
    // model.train();
}
init();
var modelOptionSelect = $('select');
modelOption = 1;
modelOptionSelect.on('change', async (e) => {
    modelOption = parseInt(e.target.value);
});

$('#toggle_visor').on('change', () => {
    tfvis.visor().toggle();
})

$('#btn_train').on('click', async () => {
    switch(modelOption){
        case 1:
                $('#btn_train').addClass('disabled');
                M.toast({html: 'Training Started!'});
                console.log('Training custom models with TFIDF features');
                await model.train();
                break;
        default:
                break;
    }
});

let model_loaded;
$('#btn_load').on('click', async () => {
    switch(modelOption){
        case 1: 
                console.log("Loading TF.js trained model");
                model_loaded = await model.load();
                model_loaded.summary();
                break;
        default:
                break;
    }
    $('#btn_load').addClass('disabled');
    $('#btn_predict').removeClass('disabled');
})

$('#btn_predict').on('click', async function () {
    const message = $('#textarea-message').val();
    if (message.trim().length <= 0){
        M.toast({ html: 'Empty message.Nothing to predict!' });
        return;
    }
    console.log("Here is the message : " + message);

    $("#chip_result").empty();
    $('#btn_predict').addClass("disabled");
    let predictedClasses = null;
    switch (modelOption) {
        case 1:
            predictedClasses = await model.predict(message, model_loaded);
            $("#chip_result").append(`Predicted Label : <div class="chip pink-text">${predictedClasses}</div>`);
            break;
           
        default:
            break;
    }

    $('#btn_predict').removeClass("disabled");

});

console.log('after')