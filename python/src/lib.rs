//! @file
//! @ingroup RNN_Wrappers
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use facaded_rnn::{
    ActivationType, CellType, LossType, RNNFacade,
    backend::BackendChoice,
};

#[pyclass]
struct PyRNNModel {
    inner: Option<RNNFacade>,
}

fn parse_cell_type(s: &str) -> PyResult<CellType> {
    s.parse::<CellType>().map_err(|e| PyValueError::new_err(e))
}

fn parse_activation(s: &str) -> PyResult<ActivationType> {
    s.parse::<ActivationType>().map_err(|e| PyValueError::new_err(e))
}

fn parse_loss(s: &str) -> PyResult<LossType> {
    s.parse::<LossType>().map_err(|e| PyValueError::new_err(e))
}

fn parse_backend(s: &str) -> PyResult<BackendChoice> {
    s.parse::<BackendChoice>().map_err(|e| PyValueError::new_err(e))
}

fn model_ref(model: &PyRNNModel) -> PyResult<&RNNFacade> {
    model.inner.as_ref().ok_or_else(|| PyValueError::new_err("Model not initialized"))
}

fn model_mut(model: &mut PyRNNModel) -> PyResult<&mut RNNFacade> {
    model.inner.as_mut().ok_or_else(|| PyValueError::new_err("Model not initialized"))
}

#[pymethods]
impl PyRNNModel {
    #[new]
    #[pyo3(signature = (
        input_size,
        hidden_sizes,
        output_size,
        cell_type = "lstm",
        activation = "tanh",
        output_activation = "linear",
        loss = "mse",
        learning_rate = 0.01,
        gradient_clip = 5.0,
        bptt_steps = 0,
        backend = "auto"
    ))]
    fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        cell_type: &str,
        activation: &str,
        output_activation: &str,
        loss: &str,
        learning_rate: f64,
        gradient_clip: f64,
        bptt_steps: usize,
        backend: &str,
    ) -> PyResult<Self> {
        let ct = parse_cell_type(cell_type)?;
        let act = parse_activation(activation)?;
        let out_act = parse_activation(output_activation)?;
        let lt = parse_loss(loss)?;
        let bc = parse_backend(backend)?;

        let mut model = RNNFacade::new(
            input_size,
            hidden_sizes,
            output_size,
            ct,
            act,
            out_act,
            lt,
            learning_rate,
            gradient_clip,
            bptt_steps,
        );

        let b = facaded_rnn::select_backend_arc(bc)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        model.set_backend(b);

        Ok(PyRNNModel { inner: Some(model) })
    }

    #[staticmethod]
    #[pyo3(signature = (filename, backend = "auto"))]
    fn load(filename: &str, backend: &str) -> PyResult<Self> {
        let mut model = RNNFacade::load_model(filename)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let bc = parse_backend(backend)?;
        let b = facaded_rnn::select_backend_arc(bc)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        model.set_backend(b);

        Ok(PyRNNModel { inner: Some(model) })
    }

    fn save(&self, filename: &str) -> PyResult<()> {
        model_ref(self)?
            .save_model(filename)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict(&mut self, inputs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(model_mut(self)?.predict(&inputs))
    }

    fn train_sequence(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> PyResult<f64> {
        Ok(model_mut(self)?.train_sequence(&inputs, &targets))
    }

    fn forward_sequence(&mut self, inputs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(model_mut(self)?.forward_sequence(&inputs))
    }

    fn backward_sequence(&mut self, targets: Vec<Vec<f64>>) -> PyResult<f64> {
        Ok(model_mut(self)?.backward_sequence(&targets))
    }

    fn reset_states(&mut self) -> PyResult<()> {
        model_mut(self)?.reset_all_states();
        Ok(())
    }

    #[pyo3(signature = (epochs, inputs, targets, verbose = false))]
    fn train(&mut self, epochs: usize, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, verbose: bool) -> PyResult<Vec<f64>> {
        let m = model_mut(self)?;
        let mut losses = Vec::with_capacity(epochs);
        for epoch in 1..=epochs {
            let loss = m.train_sequence(&inputs, &targets);
            losses.push(loss);
            if verbose && (!loss.is_nan() && !loss.is_infinite()) {
                if epoch % 10 == 0 || epoch == epochs {
                    println!("Epoch {:4}/{} - Loss: {:.6}", epoch, epochs, loss);
                }
            }
        }
        Ok(losses)
    }

    #[getter]
    fn input_size(&self) -> PyResult<usize> {
        Ok(model_ref(self)?.input_size)
    }

    #[getter]
    fn output_size(&self) -> PyResult<usize> {
        Ok(model_ref(self)?.output_size)
    }

    #[getter]
    fn hidden_sizes(&self) -> PyResult<Vec<usize>> {
        Ok(model_ref(self)?.hidden_sizes.clone())
    }

    #[getter]
    fn cell_type(&self) -> PyResult<String> {
        Ok(model_ref(self)?.cell_type.to_string())
    }

    #[getter]
    fn learning_rate(&self) -> PyResult<f64> {
        Ok(model_ref(self)?.learning_rate)
    }

    #[setter]
    fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        model_mut(self)?.learning_rate = lr;
        Ok(())
    }

    #[getter]
    fn dropout_rate(&self) -> PyResult<f64> {
        Ok(model_ref(self)?.dropout_rate)
    }

    #[setter]
    fn set_dropout_rate(&mut self, rate: f64) -> PyResult<()> {
        let m = model_mut(self)?;
        m.dropout_rate = rate;
        m.use_dropout = rate > 0.0;
        Ok(())
    }

    #[getter]
    fn gradient_clip(&self) -> PyResult<f64> {
        Ok(model_ref(self)?.gradient_clip)
    }

    #[getter]
    fn layer_count(&self) -> PyResult<usize> {
        Ok(model_ref(self)?.get_layer_count())
    }

    #[getter]
    fn sequence_length(&self) -> PyResult<usize> {
        Ok(model_ref(self)?.get_sequence_length())
    }

    #[getter]
    fn is_gpu_available(&self) -> PyResult<bool> {
        Ok(model_ref(self)?.is_gpu_available())
    }

    #[getter]
    fn backend(&self) -> PyResult<String> {
        Ok(model_ref(self)?.backend_kind().to_string())
    }

    fn get_hidden_value(&self, layer: usize, timestep: usize, neuron: usize) -> PyResult<f64> {
        Ok(model_ref(self)?.get_hidden_value(layer, timestep, neuron))
    }

    fn set_hidden_value(&mut self, layer: usize, neuron: usize, value: f64) -> PyResult<()> {
        model_mut(self)?.set_hidden_value(layer, neuron, value);
        Ok(())
    }

    fn get_output_value(&self, timestep: usize, index: usize) -> PyResult<f64> {
        Ok(model_ref(self)?.get_output_value(timestep, index))
    }

    fn get_cell_state(&self, layer: usize, neuron: usize) -> PyResult<f64> {
        Ok(model_ref(self)?.get_cell_state(layer, neuron))
    }

    fn get_gate_value(&self, layer: usize, timestep: usize, neuron: usize, gate: &str) -> PyResult<f64> {
        let gt = gate.parse::<facaded_rnn::GateType>()
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(model_ref(self)?.get_gate_value(layer, timestep, neuron, gt))
    }

    fn get_preactivation(&self, layer: usize, timestep: usize, neuron: usize) -> PyResult<f64> {
        Ok(model_ref(self)?.get_preactivation(layer, timestep, neuron))
    }

    fn get_input_value(&self, timestep: usize, index: usize) -> PyResult<f64> {
        Ok(model_ref(self)?.get_input_value(timestep, index))
    }

    fn get_sequence_outputs(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(model_ref(self)?.get_sequence_outputs())
    }

    fn get_sequence_hidden_states(&self, layer: usize) -> PyResult<Vec<Vec<f64>>> {
        Ok(model_ref(self)?.get_sequence_hidden_states(layer))
    }

    fn get_hidden_size(&self, layer: usize) -> PyResult<usize> {
        Ok(model_ref(self)?.get_hidden_size(layer))
    }

    fn detect_vanishing_gradients(&self, threshold: f64) -> PyResult<(i32, f64)> {
        Ok(model_ref(self)?.detect_vanishing_gradients(threshold))
    }

    fn detect_exploding_gradients(&self, threshold: f64) -> PyResult<(i32, f64)> {
        Ok(model_ref(self)?.detect_exploding_gradients(threshold))
    }

    fn __repr__(&self) -> PyResult<String> {
        let m = model_ref(self)?;
        Ok(format!(
            "RNNModel(input={}, hidden={:?}, output={}, cell={}, backend={})",
            m.input_size, m.hidden_sizes, m.output_size, m.cell_type, m.backend_kind()
        ))
    }
}

#[pymodule]
fn facaded_rnn_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRNNModel>()?;
    Ok(())
}

