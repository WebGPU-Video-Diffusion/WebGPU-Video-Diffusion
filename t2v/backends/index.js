import * as ORT from 'onnxruntime-web/webgpu';
import { replaceTensors } from '../util/Tensor.js';

const ONNX = ORT.default ?? ORT;

const isNode = typeof process !== 'undefined' && process?.release?.name === 'node';

const onnxSessionOptions = isNode
  ? {
    executionProviders: ['cpu'],
    executionMode: 'parallel',
  }
  : {
    executionProviders: ['webgpu'],
  };

export class Session {
  constructor(session, config = {}) {
    this.session = session;
    this.config = config || {};
  }

  static async create(modelOrPath, weightsPathOrBuffer, weightsFilename, config = {}, options = {}) {
    const arg = typeof modelOrPath === 'string' ? modelOrPath : new Uint8Array(modelOrPath);

    const sessionOptions = {
      ...onnxSessionOptions,
      ...options,
    };

    const weightsParams = {
      externalWeights: weightsPathOrBuffer,
      externalWeightsFilename: weightsFilename,
    };

    const executionProviders = (sessionOptions.executionProviders || []).map((provider) => {
      if (typeof provider === 'string') {
        return {
          name: provider,
          ...weightsParams,
        };
      }

      return {
        ...provider,
        ...weightsParams,
      };
    });

    // const session = await ONNX.InferenceSession.create(arg, {
    //   ...sessionOptions,
    //   executionProviders,
    // });
    
    //TODO: inference session currently use options directly from main, ohterwise error occurs
    const session = await ONNX.InferenceSession.create(arg, options);
    return new Session(session, config);
  }

  async run(inputs) {
    const result = await this.session.run(inputs);
    return replaceTensors(result);
  }

  release() {
    return this.session.release();
  }
}
