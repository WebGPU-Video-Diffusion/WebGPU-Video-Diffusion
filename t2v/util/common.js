import * as ort from 'onnxruntime-web/webgpu';
/*
 * initialize latents with random noise
 */
export function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

/*
 * scale the latents
*/
export function scale_model_inputs(t, sigma) {
    const d_i = t.data;
    const d_o = new Float32Array(d_i.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < d_i.length; i++) {
        d_o[i] = d_i[i] / divi;
    }
    return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
export function eulera_step(model_output, sample, sigma, gamma, vae_scaling_factor) {
    const d_o = new Float32Array(model_output.data.length);
    const prev_sample = new ort.Tensor(d_o, model_output.dims);
    const sigma_hat = sigma * (gamma + 1);

    for (let i = 0; i < model_output.data.length; i++) {
        const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
        const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
        const dt = 0 - sigma_hat;
        d_o[i] = (sample.data[i] + derivative * dt) / vae_scaling_factor;
    }
    return prev_sample;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
*/
export function draw_image(t, image_nr) {
    let pix = t.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5
        if (x < 0.) x = 0.;
        if (x > 1.) x = 1.;
        pix[i] = x;
    }
    const imageData = t.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_${image_nr}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`img_div_${image_nr}`);
    div.style.opacity = 1.
}

/*
 * get configuration from url
*/
export function toBigInt64Array(ids) {
    const src = Array.isArray(ids) ? ids : Array.from(ids);
    const out = new BigInt64Array(src.length);
    for (let i = 0; i < src.length; i++) {
        out[i] = BigInt(src[i]);
    }
    return out;
}