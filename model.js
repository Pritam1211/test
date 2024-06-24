const tf =  require('@tensorflow/tfjs');
const axios =  require('axios');
require("tfjs-node-save");
const use = require('@tensorflow-models/universal-sentence-encoder');

let modelClasifier = null;
let model = null;
console.log(tf.env().getBool('IS_NODE'))
console.log(tf.env().getBool('IS_BROWSER'))
const categories = [
  "number_sharing",
  "call_request",
  "shared_info",
  "other",
  "photo_request"
];

const trainingData = [
  ["Please call me on this number", 0],
  ["Here is my phone number", 0],
  ["WhatsApp me on this number", 0],
  ["Contact me on this phone number", 0],
  ["Call me at this number", 0],
  ["My phone number is", 0],
  ["Send me your phone number", 0],
  ["Send me your WhatsApp number", 0],
  ["My WhatsApp number is", 0],
  ["Yes, let's play Ludo", 1],
  ["Call me", 1],
  ["I agree to call", 1],
  ["Let's play Ludo together", 1],
  ["Let's play a game of Ludo", 1],
  ["I want to play Ludo", 1],
  ["I will call you", 1],
  ["Shall we play Ludo", 1],
  ["Do you want to play Ludo", 1],
  ["Call me to play Ludo", 1],
  ["My name is",2],
  ["I live in",2],
  ["My address is",2],
  ["I am from this place",2],
  ["My location is",2],
  ["My village is",2],
  ["My city is",2],
  ["I am from",2],
  ["My name is Ramesh",2],
  ["General conversation", 3],
  ["Chit chat", 3],
  ["Talking about random things", 3],
  ["What are you doing", 3],
  ["How are you", 3],
  ["Hello", 3],
  ["Hi", 3],
  ["Ok", 3],
  ["No", 3],
  ["Yes", 3],
  ["What's up", 3],
  ["How's it going", 3],
  ["send me Your photos", 4],
  ["Give me your photos", 4],
  ["Share me your photos", 4],
  ["I want to share my photos", 4],
  ["video call me", 4],
];


const trainModel = async () => {
  const model = await use.load();
  const sentences = trainingData.map(d => d[0]);
  const labels = trainingData.map(d => d[1]);
  
  const embeddings = await model.embed(sentences);
  const xs = embeddings.arraySync();
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), categories.length);

  const denseUnits = 16;
  const outputUnits = categories.length;
  const learningRate = 0.01;

  const modelTf = tf.sequential();
  modelTf.add(tf.layers.dense({ inputShape: [xs[0].length], units: denseUnits, activation: 'relu' }));
  modelTf.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));
  modelTf.compile({ optimizer: tf.train.adam(learningRate), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  await modelTf.fit(tf.tensor2d(xs), ys, { epochs: 50 });
  await modelTf.save("file://model-1.json");
  return modelTf;
};

const classifyMessage = async (sentence) => {
  const sentenceEmbedding = await model.embed([sentence]);
  const prediction = modelClasifier.predict(sentenceEmbedding);
  const categoryIndex = prediction.argMax(-1).dataSync()[0];
  return categories[categoryIndex];
};


async function translateText(text) {
  try {
    const response = await axios.post('https://deep-translator-api.azurewebsites.net/google', {
      text,
      source: "auto",
      target: "en",
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    return response.data.translation;
  } catch (error) {
    console.error('Error translating text:', error);
    return text
  }
}

// trainModel();

const test = async () =>  {
  const messages = [
        "मुझे 9876543210 पर कॉल करें",   //Hindi: "Call me on 9876543210"
        "हाँ, लूडो खेलते हैं", //"Yes, let's play Ludo"
        "മൈ നെയിം ഇസ് രമേഷ്",   //Malayalam: "My name is Ramesh"
        "నాకు కాల్ చేయండి",   //Telugu: "Call me"
        "நீங்கள் என்ன செய்கிறீர்கள்?",   //Tamil: "What are you doing?"
        "आप क्या कर रहे हो?"   //Hindi: "What are you doing?"
      ]

      // await trainModel();
      modelClasifier = await tf.loadLayersModel('file://model-1/model.json');
      model = await use.load();
       await Promise.all(messages.map(async (it, i) => {
        try {
          let msg = await translateText(it);
          const category = await classifyMessage(msg);
          console.log(`Text: ${it}   Translated: ${msg} Category: ${category}`);
          return category;
        } catch (err) {
          console.error('Error translating text:', err);
          return null;
        }

      }))
}

test();



