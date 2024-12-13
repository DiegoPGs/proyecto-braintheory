import time
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm

def optimized_training(
    experiment: str,
    regions: List[int],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[float, int, str]],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[pd.DataFrame, float]:
    """
    Versión optimizada de la función de entrenamiento para modelos LSTM.
    
    Args:
        experiment (str): Nombre del experimento ('GAMBLING', 'MOTOR', etc.)
        regions (List[int]): Lista de índices de regiones cerebrales a utilizar
        model_params (Dict): Parámetros del modelo como:
            - n_features: Número de características de entrada
            - n_timesteps: Número de pasos temporales
            - n_hidden: Dimensiones del estado oculto
            - n_layers: Número de capas LSTM
        training_params (Dict): Parámetros de entrenamiento como:
            - lr: Learning rate
            - batch_size: Tamaño del batch
            - n_epochs: Número de épocas
            - label_type: Tipo de etiqueta a predecir
        device (torch.device): Dispositivo para entrenar (CPU/GPU)
    
    Returns:
        Tuple[pd.DataFrame, float]: DataFrame con métricas de entrenamiento y tiempo total
    """
    # Inicializar temporizador
    start_time = time.time()
    
    # Inicializar modelo y moverlo al dispositivo correcto
    if training_params['label_type'] == 'flanker':
        model = LSTMRegression2(**model_params).to(device)
    else:
        model = LSTMRegression(**model_params).to(device)
    
    # Inicializar optimizador y criterio
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])
    criterion = nn.MSELoss()
    
    # Inicializar scaler para precisión mixta
    scaler = GradScaler()
    
    # Cargar y preparar datos
    X, y = load_and_prepare_data(experiment, regions, training_params['label_type'])
    
    # Convertir datos a tensores y moverlos a GPU si está disponible
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Dividir en train y test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Crear DataLoader para procesamiento por lotes eficiente
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=training_params['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    
    # Listas para almacenar métricas
    metrics = []
    
    # Loop principal de entrenamiento
    for epoch in tqdm(range(training_params['n_epochs']), desc="Entrenamiento"):
        model.train()
        train_loss = 0
        
        # Training loop
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Usar precisión mixta para acelerar el entrenamiento en GPU
            with autocast():
                output = model(batch_X)
                loss = criterion(output.view(-1), batch_y)
                
                if training_params['label_type'] in ['WL', 'gender']:
                    accuracy = ((output.view(-1) > 0.5) == batch_y).float().mean()
            
            # Backward pass con scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            y_hat = model(X_test)
            test_loss = criterion(y_hat.view(-1), y_test)
            
            if training_params['label_type'] in ['WL', 'gender']:
                test_accuracy = ((y_hat.view(-1) > 0.5) == y_test).float().mean()
        
        # Guardar métricas
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'test_loss': test_loss.item()
        }
        
        if training_params['label_type'] in ['WL', 'gender']:
            metrics_dict.update({
                'train_accuracy': accuracy.item(),
                'test_accuracy': test_accuracy.item()
            })
            
        metrics.append(metrics_dict)
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    
    return pd.DataFrame(metrics), total_time
