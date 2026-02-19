//! @file
//! @ingroup RNN_Wrappers
use napi::bindgen_prelude::*;
use napi_derive::napi;

use facaded_rnn::{
    ActivationType, CellType, LossType, RNNFacade,
    backend::BackendChoice,
};

fn to_napi_err<E: std::fmt::Display>(e: E) -> Error {
    Error::from_reason(e.to_string())
}

#[napi(object)]
pub struct RNNModelOptions {
    pub input_size: u32,
    pub hidden_sizes: Vec<u32>,
    pub output_size: u32,
    pub cell_type: Option<String>,
    pub activation: Option<String>,
    pub output_activation: Option<String>,
    pub loss: Option<String>,
    pub learning_rate: Option<f64>,
    pub gradient_clip: Option<f64>,
    pub bptt_steps: Option<u32>,
    pub backend: Option<String>,
}

#[napi(object)]
pub struct TrainOptions {
    pub epochs: u32,
    pub verbose: Option<bool>,
}

#[napi(object)]
pub struct ModelInfo {
    pub input_size: u32,
    pub output_size: u32,
    pub hidden_sizes: Vec<u32>,
    pub cell_type: String,
    pub layer_count: u32,
    pub learning_rate: f64,
    pub gradient_clip: f64,
    pub dropout_rate: f64,
    pub sequence_length: u32,
    pub is_gpu_available: bool,
    pub backend: String,
}

#[napi(object)]
pub struct GradientDiagnostic {
    pub count: i32,
    pub value: f64,
}

#[napi]
pub struct RNNModel {
    inner: RNNFacade,
}

#[napi]
impl RNNModel {
    #[napi(constructor)]
    pub fn new(options: RNNModelOptions) -> Result<Self> {
        let cell_type: CellType = options.cell_type.as_deref().unwrap_or("lstm")
            .parse().map_err(to_napi_err)?;
        let activation: ActivationType = options.activation.as_deref().unwrap_or("tanh")
            .parse().map_err(to_napi_err)?;
        let output_activation: ActivationType = options.output_activation.as_deref().unwrap_or("linear")
            .parse().map_err(to_napi_err)?;
        let loss: LossType = options.loss.as_deref().unwrap_or("mse")
            .parse().map_err(to_napi_err)?;
        let backend_choice: BackendChoice = options.backend.as_deref().unwrap_or("auto")
            .parse().map_err(to_napi_err)?;

        let lr = options.learning_rate.unwrap_or(0.01);
        let clip = options.gradient_clip.unwrap_or(5.0);
        let bptt = options.bptt_steps.unwrap_or(0) as usize;

        let hidden: Vec<usize> = options.hidden_sizes.iter().map(|&h| h as usize).collect();

        let mut model = RNNFacade::new(
            options.input_size as usize,
            hidden,
            options.output_size as usize,
            cell_type,
            activation,
            output_activation,
            loss,
            lr,
            clip,
            bptt,
        );

        let b = facaded_rnn::select_backend_arc(backend_choice).map_err(to_napi_err)?;
        model.set_backend(b);

        Ok(RNNModel { inner: model })
    }

    #[napi(factory)]
    pub fn load(filename: String, backend: Option<String>) -> Result<Self> {
        let mut model = RNNFacade::load_model(&filename).map_err(to_napi_err)?;

        let bc: BackendChoice = backend.as_deref().unwrap_or("auto")
            .parse().map_err(to_napi_err)?;
        let b = facaded_rnn::select_backend_arc(bc).map_err(to_napi_err)?;
        model.set_backend(b);

        Ok(RNNModel { inner: model })
    }

    #[napi]
    pub fn save(&self, filename: String) -> Result<()> {
        self.inner.save_model(&filename).map_err(to_napi_err)
    }

    #[napi]
    pub fn predict(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.inner.predict(&inputs)
    }

    #[napi]
    pub fn train_sequence(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
        self.inner.train_sequence(&inputs, &targets)
    }

    #[napi]
    pub fn forward_sequence(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.inner.forward_sequence(&inputs)
    }

    #[napi]
    pub fn backward_sequence(&mut self, targets: Vec<Vec<f64>>) -> f64 {
        self.inner.backward_sequence(&targets)
    }

    #[napi]
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, options: TrainOptions) -> Vec<f64> {
        let epochs = options.epochs as usize;
        let verbose = options.verbose.unwrap_or(false);
        let mut losses = Vec::with_capacity(epochs);

        for epoch in 1..=epochs {
            let loss = self.inner.train_sequence(&inputs, &targets);
            losses.push(loss);
            if verbose && !loss.is_nan() && !loss.is_infinite() {
                if epoch % 10 == 0 || epoch == epochs {
                    println!("Epoch {:4}/{} - Loss: {:.6}", epoch, epochs, loss);
                }
            }
        }

        losses
    }

    #[napi]
    pub fn reset_states(&mut self) {
        self.inner.reset_all_states();
    }

    #[napi]
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            input_size: self.inner.input_size as u32,
            output_size: self.inner.output_size as u32,
            hidden_sizes: self.inner.hidden_sizes.iter().map(|&h| h as u32).collect(),
            cell_type: self.inner.cell_type.to_string(),
            layer_count: self.inner.get_layer_count() as u32,
            learning_rate: self.inner.learning_rate,
            gradient_clip: self.inner.gradient_clip,
            dropout_rate: self.inner.dropout_rate,
            sequence_length: self.inner.get_sequence_length() as u32,
            is_gpu_available: self.inner.is_gpu_available(),
            backend: self.inner.backend_kind().to_string(),
        }
    }

    #[napi(getter)]
    pub fn input_size(&self) -> u32 {
        self.inner.input_size as u32
    }

    #[napi(getter)]
    pub fn output_size(&self) -> u32 {
        self.inner.output_size as u32
    }

    #[napi(getter)]
    pub fn hidden_sizes(&self) -> Vec<u32> {
        self.inner.hidden_sizes.iter().map(|&h| h as u32).collect()
    }

    #[napi(getter)]
    pub fn cell_type(&self) -> String {
        self.inner.cell_type.to_string()
    }

    #[napi(getter)]
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }

    #[napi(setter)]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.learning_rate = lr;
    }

    #[napi(getter)]
    pub fn dropout_rate(&self) -> f64 {
        self.inner.dropout_rate
    }

    #[napi(setter)]
    pub fn set_dropout_rate(&mut self, rate: f64) {
        self.inner.dropout_rate = rate;
        self.inner.use_dropout = rate > 0.0;
    }

    #[napi(getter)]
    pub fn gradient_clip(&self) -> f64 {
        self.inner.gradient_clip
    }

    #[napi(getter)]
    pub fn layer_count(&self) -> u32 {
        self.inner.get_layer_count() as u32
    }

    #[napi(getter)]
    pub fn sequence_length(&self) -> u32 {
        self.inner.get_sequence_length() as u32
    }

    #[napi(getter)]
    pub fn is_gpu_available(&self) -> bool {
        self.inner.is_gpu_available()
    }

    #[napi(getter)]
    pub fn backend(&self) -> String {
        self.inner.backend_kind().to_string()
    }

    #[napi]
    pub fn get_hidden_value(&self, layer: u32, timestep: u32, neuron: u32) -> f64 {
        self.inner.get_hidden_value(layer as usize, timestep as usize, neuron as usize)
    }

    #[napi]
    pub fn set_hidden_value(&mut self, layer: u32, neuron: u32, value: f64) {
        self.inner.set_hidden_value(layer as usize, neuron as usize, value);
    }

    #[napi]
    pub fn get_output_value(&self, timestep: u32, index: u32) -> f64 {
        self.inner.get_output_value(timestep as usize, index as usize)
    }

    #[napi]
    pub fn get_cell_state(&self, layer: u32, neuron: u32) -> f64 {
        self.inner.get_cell_state(layer as usize, neuron as usize)
    }

    #[napi]
    pub fn get_gate_value(&self, layer: u32, timestep: u32, neuron: u32, gate: String) -> Result<f64> {
        let gt: facaded_rnn::GateType = gate.parse().map_err(to_napi_err)?;
        Ok(self.inner.get_gate_value(layer as usize, timestep as usize, neuron as usize, gt))
    }

    #[napi]
    pub fn get_preactivation(&self, layer: u32, timestep: u32, neuron: u32) -> f64 {
        self.inner.get_preactivation(layer as usize, timestep as usize, neuron as usize)
    }

    #[napi]
    pub fn get_input_value(&self, timestep: u32, index: u32) -> f64 {
        self.inner.get_input_value(timestep as usize, index as usize)
    }

    #[napi]
    pub fn get_sequence_outputs(&self) -> Vec<Vec<f64>> {
        self.inner.get_sequence_outputs()
    }

    #[napi]
    pub fn get_sequence_hidden_states(&self, layer: u32) -> Vec<Vec<f64>> {
        self.inner.get_sequence_hidden_states(layer as usize)
    }

    #[napi]
    pub fn get_hidden_size(&self, layer: u32) -> u32 {
        self.inner.get_hidden_size(layer as usize) as u32
    }

    #[napi]
    pub fn detect_vanishing_gradients(&self, threshold: f64) -> GradientDiagnostic {
        let (count, value) = self.inner.detect_vanishing_gradients(threshold);
        GradientDiagnostic { count, value }
    }

    #[napi]
    pub fn detect_exploding_gradients(&self, threshold: f64) -> GradientDiagnostic {
        let (count, value) = self.inner.detect_exploding_gradients(threshold);
        GradientDiagnostic { count, value }
    }
}

