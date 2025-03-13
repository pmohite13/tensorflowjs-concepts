import * as tf from "@tensorflow/tfjs";
// import "@tensorflow/tfjs-backend-wasm";
import * as model from './model';

console.log(tf.version);

console.log(tf.getBackend())
// tf.setBackend('wasm');

// tf.ready().then(() => {
//     console.log(tf.getBackend())
// }) 

const init = async () => {
    await tf.ready();
    console.log(tf.getBackend());
    model.train();
}
init();
console.log('after')