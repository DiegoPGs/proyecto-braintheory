class HCPDataset:
    """
    Clase para manejar la carga y preprocesamiento de datos del HCP.
    
    Esta clase proporciona una interfaz unificada para trabajar con los datos
    del Human Connectome Project, incluyendo la carga de series temporales,
    variables explicativas (EVs) y datos demográficos.
    """
    
    def __init__(self, root_dir: str, experiment: str):
        """
        Inicializa el dataset HCP.
        
        Args:
            root_dir (str): Directorio raíz donde se encuentran los datos
            experiment (str): Nombre del experimento a analizar
        """
        self.root_dir = root_dir
        self.experiment = experiment
        self.subjects = np.loadtxt(
            os.path.join(root_dir, 'subjects_list.txt'), 
            dtype='str'
        )
        self._load_regions()
        
    def _load_regions(self) -> None:
        """Carga información sobre las regiones cerebrales."""
        regions = np.load(f"{self.root_dir}/regions.npy").T
        self.region_info = {
            'name': regions[0].tolist(),
            'network': regions[1],
            'hemi': ['Right']*180 + ['Left']*180
        }
    
    def load_single_timeseries(
        self, 
        subject: str, 
        run: int, 
        remove_mean: bool = True
    ) -> np.ndarray:
        """
        Carga series temporales para un sujeto y run específicos.
        
        Args:
            subject (str): ID del sujeto
            run (int): Número de run (0 o 1)
            remove_mean (bool): Si se debe remover la media de cada parcela
            
        Returns:
            np.ndarray: Matriz de datos BOLD (n_parcelas x n_timepoints)
        """
        bold_run = CONSTANTS['RUNS'][run]
        bold_path = os.path.join(
            self.root_dir, 
            'subjects', 
            subject, 
            self.experiment,
            f"tfMRI_{self.experiment}_{bold_run}"
        )
        ts = np.load(os.path.join(bold_path, "data.npy"))
        
        if remove_mean:
            ts -= ts.mean(axis=1, keepdims=True)
            
        return ts
    
    def load_evs(self, subject: str, run: int) -> List[List]:
        """
        Carga variables explicativas (EVs) para un sujeto y run.
        
        Args:
            subject (str): ID del sujeto
            run (int): Número de run (0 o 1)
            
        Returns:
            List[List]: Lista de frames asociados con cada condición
        """
        frames_list = []
        task_key = f'tfMRI_{self.experiment}_{CONSTANTS["RUNS"][run]}'
        
        for cond in EXPERIMENTS[self.experiment]['cond']:
            ev_file = os.path.join(
                self.root_dir, 
                'subjects', 
                subject,
                self.experiment,
                task_key,
                'EVs',
                f"{cond}.txt"
            )
            
            # Cargar y procesar EV
            ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
            ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
            
            # Convertir tiempos a frames
            start = np.floor(ev["onset"] / CONSTANTS['TR']).astype(int)
            duration = np.ceil(ev["duration"] / CONSTANTS['TR']).astype(int)
            frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
            
            frames_list.append(frames)
            
        return frames_list
    
    def get_preprocessed_data(
        self, 
        subjects: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene datos preprocesados listos para entrenamiento.
        
        Args:
            subjects (List[str], optional): Lista de sujetos a procesar.
                Si es None, usa todos los sujetos.
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, y) datos y etiquetas
        """
        # Implementar lógica de preprocesamiento aquí
        pass