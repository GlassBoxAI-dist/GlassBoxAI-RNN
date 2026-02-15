use facaded_rnn::{
    ActivationType, CellType, LossType, RNNFacade,
    backend::BackendChoice,
    load_data_from_csv, select_backend_arc,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "facaded_rnn")]
#[command(about = "Facaded RNN with GPU acceleration (CUDA/OpenCL/CPU)", long_about = None)]
struct Cli {
    #[arg(long, default_value = "auto", global = true)]
    backend: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Create {
        #[arg(long)]
        input: usize,
        #[arg(long, value_delimiter = ',')]
        hidden: Vec<usize>,
        #[arg(long)]
        output: usize,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "lstm")]
        cell: String,
        #[arg(long, default_value = "0.01")]
        lr: f64,
        #[arg(long, default_value = "tanh")]
        hidden_act: String,
        #[arg(long, default_value = "linear")]
        output_act: String,
        #[arg(long, default_value = "mse")]
        loss: String,
        #[arg(long, default_value = "5.0")]
        clip: f64,
        #[arg(long, default_value = "0")]
        bptt: usize,
    },
    Train {
        #[arg(long)]
        model: String,
        #[arg(long)]
        data: String,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "100")]
        epochs: usize,
        #[arg(long, default_value = "1")]
        batch: usize,
        #[arg(long)]
        lr: Option<f64>,
        #[arg(long)]
        seq_len: Option<usize>,
        #[arg(long, default_value = "false")]
        verbose: bool,
    },
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long, value_delimiter = ',')]
        input: Vec<f64>,
    },
    Info {
        #[arg(long)]
        model: String,
    },
    Query {
        #[arg(long)]
        model: String,
        #[arg(long)]
        query_type: String,
        #[arg(long, default_value = "0")]
        layer: usize,
        #[arg(long, default_value = "0")]
        timestep: usize,
        #[arg(long, default_value = "0")]
        neuron: usize,
        #[arg(long, default_value = "0")]
        index: usize,
        #[arg(long)]
        gate: Option<String>,
        #[arg(long)]
        dropout_rate: Option<f64>,
        #[arg(long, default_value = "false")]
        enable_dropout: bool,
        #[arg(long, default_value = "false")]
        disable_dropout: bool,
    },
    Help,
}

fn print_usage() {
    println!("Facaded RNN with GPU acceleration (CUDA/OpenCL/CPU)\n");
    println!("Global Options:");
    println!("  --backend=TYPE         auto|cpu|cuda|opencl|hybrid (default: auto)\n");
    println!("Commands:");
    println!("  create   Create a new RNN model and save to JSON");
    println!("  train    Train an existing model with data from JSON");
    println!("  predict  Make predictions with a trained model from JSON");
    println!("  info     Display model information from JSON");
    println!("  query    Query model state and internals (facade functions)");
    println!("  help     Show this help message\n");
    println!("Create Options:");
    println!("  --input=N              Input layer size (required)");
    println!("  --hidden=N,N,...       Hidden layer sizes (required)");
    println!("  --output=N             Output layer size (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)");
    println!("  --cell=TYPE            simplernn|lstm|gru (default: lstm)");
    println!("  --lr=VALUE             Learning rate (default: 0.01)");
    println!("  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)");
    println!("  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)");
    println!("  --loss=TYPE            mse|crossentropy (default: mse)");
    println!("  --clip=VALUE           Gradient clipping (default: 5.0)");
    println!("  --bptt=N               BPTT steps (default: 0 = full)\n");
    println!("Train Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Training data CSV file (required)");
    println!("  --save=FILE.json       Save trained model to JSON (required)");
    println!("  --epochs=N             Number of training epochs (default: 100)");
    println!("  --batch=N              Batch size (default: 1)");
    println!("  --lr=VALUE             Override learning rate");
    println!("  --seq-len=N            Sequence length (default: auto-detect)\n");
    println!("Predict Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --input=v1,v2,...      Input values as CSV (required)\n");
    println!("Info Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)\n");
    println!("Query Options (Facade Functions):");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --query-type=TYPE      Query type (required)");
    println!("                         Valid types: input-size, output-size, hidden-size,");
    println!("                                      cell-type, sequence-length, dropout-rate,");
    println!("                                      hidden-state");
    println!("  --layer=N              Layer index");
    println!("  --timestep=N           Timestep index");
    println!("  --neuron=N             Neuron index");
    println!("  --index=N              Generic index parameter");
    println!("  --dropout-rate=VALUE   Set dropout rate (0.0-1.0)");
    println!("  --enable-dropout       Enable dropout");
    println!("  --disable-dropout      Disable dropout\n");
    println!("Examples:");
    println!("  facaded_rnn create --input=2 --hidden=16 --output=2 --cell=lstm --save=seq.json");
    println!("  facaded_rnn train --model=seq.json --data=seq.csv --epochs=200 --save=seq_trained.json");
    println!("  facaded_rnn predict --model=seq_trained.json --input=0.5,0.5");
    println!("  facaded_rnn info --model=seq_trained.json");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let backend_choice: BackendChoice = cli.backend.parse()
        .map_err(|e: String| -> Box<dyn std::error::Error> { e.into() })?;
    let backend = select_backend_arc(backend_choice)?;
    println!("Backend: {}", backend.kind());

    match cli.command {
        Commands::Create {
            input,
            hidden,
            output,
            save,
            cell,
            lr,
            hidden_act,
            output_act,
            loss,
            clip,
            bptt,
        } => {
            let cell_type: CellType = cell.parse()?;
            let hidden_activation: ActivationType = hidden_act.parse()?;
            let output_activation: ActivationType = output_act.parse()?;
            let loss_type: LossType = loss.parse()?;

            let mut model = RNNFacade::new(
                input,
                hidden.clone(),
                output,
                cell_type,
                hidden_activation,
                output_activation,
                loss_type,
                lr,
                clip,
                bptt,
            );

            model.set_backend(backend);

            println!("Created RNN model:");
            println!("  Input size: {}", input);
            println!("  Hidden sizes: {:?}", hidden);
            println!("  Output size: {}", output);
            println!("  Cell type: {}", cell_type);
            println!("  Hidden activation: {}", hidden_activation);
            println!("  Output activation: {}", output_activation);
            println!("  Loss function: {}", loss_type);
            println!("  Learning rate: {:.6}", lr);
            println!("  Gradient clip: {:.2}", clip);
            println!("  BPTT steps: {}", bptt);
            println!("  GPU Available: {}", if model.is_gpu_available() { "Yes" } else { "No" });

            model.save_model(&save)?;
            println!("Model saved to JSON: {}", save);
        }

        Commands::Train {
            model,
            data,
            save,
            epochs,
            batch: _,
            lr,
            seq_len: _,
            verbose,
        } => {
            println!("Loading model from JSON: {}", model);
            let mut rnn_model = RNNFacade::load_model(&model)?;

            if let Some(new_lr) = lr {
                rnn_model.learning_rate = new_lr;
            }

            rnn_model.set_backend(backend);

            println!("Model loaded successfully.");

            println!("Loading training data from: {}", data);
            let (inputs, targets) = load_data_from_csv(&data)?;

            if inputs.is_empty() {
                return Err("No data loaded from CSV file".into());
            }

            println!("Loaded {} timesteps of training data", inputs.len());
            println!("Starting training for {} epochs...", epochs);

            for epoch in 1..=epochs {
                let train_loss = rnn_model.train_sequence(&inputs, &targets);

                if !train_loss.is_nan() && !train_loss.is_infinite()
                    && (verbose || (epoch % 10 == 0) || (epoch == epochs))
                {
                    println!(
                        "Epoch {:4}/{} - Loss: {:.6}",
                        epoch, epochs, train_loss
                    );
                }
            }

            println!("Training completed.");
            println!("Saving trained model to: {}", save);
            rnn_model.save_model(&save)?;
            println!("Model saved to JSON: {}", save);
        }

        Commands::Predict { model, input } => {
            let mut rnn_model = RNNFacade::load_model(&model)?;
            rnn_model.set_backend(backend);

            let inputs = vec![input.clone()];
            let predictions = rnn_model.predict(&inputs);

            print!("Input: ");
            for (i, v) in input.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.4}", v);
            }
            println!();

            if !predictions.is_empty() && !predictions.last().unwrap().is_empty() {
                print!("Output: ");
                let last_pred = predictions.last().unwrap();
                for (i, v) in last_pred.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.6}", v);
                }
                println!();

                if last_pred.len() > 1 {
                    let max_idx = last_pred
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();
                    println!("Max index: {}", max_idx);
                }
            }
        }

        Commands::Info { model } => {
            println!("Loading model from JSON: {}", model);
            let rnn_model = RNNFacade::load_model(&model)?;

            println!("Model Information:");
            println!("  Layers: {}", rnn_model.get_layer_count());
            print!("  Hidden sizes: ");
            for (i, _) in rnn_model.hidden_sizes.iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                print!("{}", rnn_model.get_hidden_size(i));
            }
            println!();
            println!("  Cell type: {}", rnn_model.cell_type);
            println!("  Learning rate: {:.6}", rnn_model.learning_rate);
            println!("  Gradient clip: {:.2}", rnn_model.gradient_clip);
            println!("  Dropout rate: {:.6}", rnn_model.dropout_rate);
            println!("  Backend: {}", backend.kind());
        }

        Commands::Query {
            model,
            query_type,
            layer,
            timestep,
            neuron,
            index: _,
            gate: _,
            dropout_rate,
            enable_dropout: _,
            disable_dropout: _,
        } => {
            println!("Loading model from JSON: {}", model);
            let mut rnn_model = RNNFacade::load_model(&model)?;
            rnn_model.set_backend(backend);

            println!("Executing query: {}\n", query_type);

            match query_type.as_str() {
                "input-size" => {
                    println!("Input size: {}", rnn_model.input_size);
                }
                "output-size" => {
                    println!("Output size: {}", rnn_model.output_size);
                }
                "hidden-size" => {
                    println!("Hidden size (layer {}): {}", layer, rnn_model.get_hidden_size(layer));
                }
                "cell-type" => {
                    println!("Cell type: {}", rnn_model.cell_type);
                }
                "sequence-length" => {
                    println!("Sequence length: {}", rnn_model.get_sequence_length());
                }
                "dropout-rate" => {
                    println!("Current dropout rate: {:.6}", rnn_model.dropout_rate);
                }
                "hidden-state" => {
                    println!(
                        "Hidden state at [{},{},{}]: {:.6}",
                        layer, timestep, neuron,
                        rnn_model.get_hidden_value(layer, timestep, neuron)
                    );
                }
                _ => {
                    println!("Unknown query type: {}", query_type);
                }
            }

            if let Some(rate) = dropout_rate {
                rnn_model.dropout_rate = rate;
                rnn_model.use_dropout = rate > 0.0;
                println!("Dropout rate set to: {:.6}", rate);
            }
        }

        Commands::Help => {
            print_usage();
        }
    }

    Ok(())
}
