from .module_base import PipelineBase, ConfigBase, DataModuleBase, Vocab, ProbePipelineBase, DirectoryHandlerBase
from typing import Optional, final
import os, time
import torch.nn as nn
from .model_bank import GPT2Standard
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from .utils import clever_load, clever_save, EasyDict, EpochCheckpointCallback
import torch
import re
from copy import deepcopy
import wandb

class TrainingManagerBase():
    """
    TrainingManagerBase is a base class for managing the training process of a machine learning model. 
    It handles the initialization of various components required for training, such as configuration, data modules, pipelines, and logging. It also provides methods for setting up modules, restoring state from checkpoints, and fitting the model.
    Attributes:
        dir_handler (DirectoryHandlerBase): Handles directory paths for loading and saving configurations, checkpoints, etc.
        abstract_config (ConfigBase): Abstract class for configuration management.
        abstract_pipeline (PipelineBase): Abstract class for pipeline management.
        abstract_datamodule (DataModuleBase): Abstract class for data module management.
        config (ConfigBase): Loaded configuration object.
        data_config (dict): Data configuration parameters.
        model_config (dict): Model configuration parameters.
        train_config (dict): Training configuration parameters.
        vocab (Vocab): Vocabulary object if vocab file exists, otherwise None.
        training_name (str): Generated name for the training run.
        wandb_logger (WandbLogger): Weights and Biases logger object if use_wandb is True, otherwise None.
        trainer (Trainer): PyTorch Lightning Trainer object for managing the training loop.
    Methods:
        __init__(dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule): 
            Initializes the TrainingManagerBase with directory handler, configuration, pipeline, and data module.
        restore_state(path_to_dirhandler, use_wandb, keep_output_dir, abstract_config, abstract_pipeline, abstract_datamodule): 
            Class method to restore the state from a saved directory handler.
        setup_modules_init(): 
            Initializes the data module and pipeline.
        setup_modules_restore(ckpt_path): 
            Restores the data module and pipeline from a checkpoint.
        wandb_initialization(use_wandb): 
            Initializes Weights and Biases logging if use_wandb is True.
        fit(): 
            Fits the model using the trainer.
        get_training_name(): 
            Generates a unique name for the training run.
        config_datamodule(): 
            Prepares keyword arguments for initializing the data module.
        config_pipeline(): 
            Prepares keyword arguments for initializing the pipeline.
    """
    def __init__(self, 
                 dir_handler: DirectoryHandlerBase,
                 abstract_config: ConfigBase = ConfigBase,
                 abstract_pipeline: PipelineBase = PipelineBase,
                 abstract_datamodule: DataModuleBase = DataModuleBase,
                 abstract_probepipeline: ProbePipelineBase = ProbePipelineBase,
                 **kwargs,
                ):
        """
        kwargs will override the configuration parameters.
        By default, you can either supply kwargs like "train_config.batch_size" or simply "batch_size" if this key is already in the configuration. Otherwise, please specify the full path of the key in the configuration.
        """
        self.dir_handler = dir_handler
        
        # set up abstract classes
        self.abstract_config = abstract_config
        self.abstract_pipeline = abstract_pipeline
        self.abstract_datamodule = abstract_datamodule
        self.abatract_probepipeline = abstract_probepipeline

        # set up configuration
        config_dir = self.dir_handler.load_config_dir
        self.config = self.abstract_config(config_dir)
        verbose = kwargs.get('verbose', False)
        self.config.override(kwargs, verbose=verbose)

        # seed_everything
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed, workers=True)
        else:
            self.train_config.seed = int(time.time())
            seed_everything(self.train_config.seed, workers=True)

        # sync vocab size and obtain vocab
        if os.path.exists(self.dir_handler.vocab_path):
            self.vocab = Vocab(clever_load(self.dir_handler.vocab_path))
            # self.model_config.vocab_size = len(self.vocab)
            self.data_config.vocab_size = len(self.vocab)
        else:
            self.vocab = None
            # self.model_config.vocab_size = None
            self.data_config.vocab_size = None
        
        # output directory, and generate a training name
        self.training_name, self.group_info = self.get_training_name()
        # set up output directory
        self.dir_handler.set_output_dir(self.training_name)
        self.config.override({'output_dir': self.dir_handler.output_dir}, verbose=verbose)
        
        self.data_config.num_workers = self.train_config.num_workers

        # setup modules
        if self.dir_handler.load_ckpt_path is not None:
            self.setup_modules_restore(self.dir_handler.load_ckpt_path)
        else: 
            self.setup_modules_init()
        
        # wandb initialization
        self.wandb_logger = self.wandb_initialization(self.train_config.use_wandb)
        # self.train_config.wandb_id = self.wandb_logger.version() if use_wandb else None

        # save configuration as a yaml file
        self.config.save_to_dir(self.dir_handler.output_config_dir)
        self.dir_handler.save_to_file(self.dir_handler.output_dirhandler_path)

    @classmethod
    def restore_state(cls, 
                      path_to_dirhandler: str,
                      abstract_config: ConfigBase = ConfigBase,
                      abstract_pipeline: PipelineBase = PipelineBase,
                      abstract_datamodule: DataModuleBase = DataModuleBase,
                      abstract_probepipeline: ProbePipelineBase = ProbePipelineBase,
                      **kwargs,
                      ):
        dir_handler = DirectoryHandlerBase.load_from_file(path_to_dirhandler)
        return cls(dir_handler,
                   abstract_config,
                   abstract_pipeline,
                   abstract_datamodule,
                   abstract_probepipeline,
                     **kwargs,
                    )
    
    @classmethod
    def load_training_manager(
        cls,
        last_run_dir: str,
        ckpt_file_path: Optional[str] = None,
        prefix_for_training_name: Optional[str] = None,
        new_training_name: Optional[str] = None,
        abstract_config: ConfigBase = ConfigBase,
        abstract_pipeline: PipelineBase = PipelineBase,
        abstract_datamodule: DataModuleBase = DataModuleBase,
        **kwargs
    ):

        last_run_name = os.path.basename(last_run_dir)
        # suppose last_run_dir has the structure 'task_dir/run/last_run_name', get the task_dir
        task_dir = os.path.dirname(os.path.dirname(last_run_dir))
        if ckpt_file_path is None:
            # search for the latest checkpoint file
            ckpt_files = [f for f in os.listdir(last_run_dir) if f.endswith('.ckpt')]
            ckpt_files.sort()
            ckpt_file_path = os.path.join(last_run_dir, ckpt_files[-1])
        if prefix_for_training_name is None:
            prefix_for_training_name = ''
        training_name = prefix_for_training_name + last_run_name
        
        if new_training_name is not None:
            training_name = new_training_name
        
        # dir_handler = DirectoryHandlerBase(
        #     load_data_abs_dir=None,
        #     data_file_name=None,
        #     vocab_file_name=None,
        #     load_config_abs_dir=os.path.join(last_run_dir, 'configurations'),
        #     load_ckpt_abs_path=ckpt_file_path,
        #     output_abs_dir=None,
        #     create_run_under_abs_dir=task_dir,
        #     training_name=training_name,
        # )

        path_to_dirhandler = os.path.join(last_run_dir, 'configurations', 'dirhandler.yaml')
        dir_handler_old = DirectoryHandlerBase.load_from_file(path_to_dirhandler)
        dir_handler_old.load_config_abs_dir = os.path.dirname(path_to_dirhandler)
        dir_handler_old.load_ckpt_abs_path = ckpt_file_path
        # dir_handler.data_file_name = dir_handler_old.data_file_name
        # dir_handler.vocab_file_name = dir_handler_old.vocab_file_name

        # get the data directory
        data_dir = os.path.basename(os.path.normpath(dir_handler_old.load_data_abs_dir)) if 'data_dir' not in kwargs else kwargs['data_dir']
        dir_handler_old.load_data_abs_dir = os.path.join(task_dir, 'data', data_dir)
        
        dir_handler_old.output_abs_dir = os.path.join(task_dir, 'run', training_name)

        return cls(
            dir_handler=dir_handler_old,
            abstract_config=abstract_config,
            abstract_pipeline=abstract_pipeline,
            abstract_datamodule=abstract_datamodule,
            **kwargs
        )
    
    @property
    def data_config(self):
        return self.config.data_config
    
    @property
    def model_config(self):
        return self.config.model_config
    
    @property
    def train_config(self):
        return self.config.train_config
    
    @property
    def probe_config(self):
        return self.config.probe_config
    
    # for other config, please call self.config.<other_config>
    
    @final
    def setup_probe_pipeline(self):
        self.probe_pipeline = self.abatract_probepipeline(**self.config_probepipeline())
        
    @final
    def load_probe_pipeline(self, ckpt_path):
        self.probe_pipeline = self.abatract_probepipeline.load_from_checkpoint(ckpt_path, **self.config_probepipeline())
    
    @final
    def setup_modules_init(self):
        self.datamodule = self.abstract_datamodule(**self.config_datamodule())
        self.pipeline = self.abstract_pipeline(**self.config_pipeline())

    @final
    def setup_modules_restore(self, ckpt_path):
        self.datamodule = self.abstract_datamodule(**self.config_datamodule())
        self.pipeline = self.abstract_pipeline.load_from_checkpoint(ckpt_path, **self.config_pipeline())
        # state_dict = torch.load(ckpt_path)['state_dict']
        # self.pipeline.load_state_dict(state_dict, strict=False)


    @final
    def wandb_initialization(self, use_wandb):
        if use_wandb:
            wandb_config = self.train_config.wandb_config
                    # Reinitialize Wandb (this creates a new run each time)
            if self.group_info == None:
                wandb.init(reinit=True,  # Reinitialize the run for each call
                        project=wandb_config.wandb_project,
                        entity=wandb_config.wandb_entity,
                        name=self.dir_handler.training_name
                        )
            else:
                wandb.init(reinit=True,  # Reinitialize the run for each call
                        project=wandb_config.wandb_project,
                        entity=wandb_config.wandb_entity,
                        group=self.group_info[0], 
                        name = self.group_info[1]
                        )

            wandb_logger = WandbLogger(
                    name=self.dir_handler.training_name,
                    project=wandb_config.wandb_project,
                    save_dir=self.dir_handler.output_dir,
                    entity=wandb_config.wandb_entity,
                )
            config_copy = deepcopy(self.config.to_dict())
            config_copy.update({'dir_handler': self.dir_handler.__dict__})
            wandb_logger.log_hyperparams(config_copy)
        else:
            wandb_logger = False
        return wandb_logger
    
    def generate_customed_trainer(self, pipeline_type='train', callbacks_ls=[], **kwargs):
        if pipeline_type == 'train':
            pipeline = self.pipeline
        elif pipeline_type == 'probe':
            pipeline = self.probe_pipeline
        
        callback_ls = []
        for cb in callbacks_ls:
            if cb.endswith('_ckpt'):
                if cb == 'min_val_loss_ckpt':
                    callback_ls.append(ModelCheckpoint(
                        dirpath=self.dir_handler.output_dir,
                        filename='{epoch}-{val_loss:.4f}', 
                        monitor='val_loss',
                        mode='min',
                    ))
                # check if the callback name is 'per_{n}_epoch_ckpt' where n is an integer
                elif re.match(r'per_\d+_epoch_ckpt', cb):
                    n = int(cb.split('_')[1])
                    callback_ls.append(ModelCheckpoint(
                        dirpath=self.dir_handler.output_dir,
                        filename='{epoch}-{val_loss:.4f}', 
                        monitor='val_loss',
                        every_n_epochs=n, 
                        save_top_k=-1,
                    ))
                elif cb == 'per_epoch_ckpt':
                    callback_ls.append(ModelCheckpoint(
                        dirpath=self.dir_handler.output_dir,
                        filename='{epoch}-{val_loss:.4f}', 
                        monitor='val_loss',
                        every_n_epochs=1, 
                        save_top_k=-1,
                    ))
                # check if the callback name is 'epoch_[a, b, c...]_ckpt' where a, b, c... is a list of integers
                elif re.match(r'epoch_\[.*\]_ckpt', cb):
                    ckpt_epochs = [int(i) for i in cb.split('_')[1][1:-1].split(',')]
                    callback_ls.append(EpochCheckpointCallback(
                        ckpt_epochs=ckpt_epochs,
                        dirpath=self.dir_handler.output_dir,
                    ))
                # check if the callback name is 'top_{k}_ckpt' where k is an integer
                elif re.match(r'top_\d+_ckpt', cb):
                    k = int(cb.split('_')[1])
                    callback_ls.append(ModelCheckpoint(
                        dirpath=self.dir_handler.output_dir,
                        filename='{epoch}-{val_loss:.4f}', 
                        monitor='val_loss',
                        save_top_k=k,
                    ))
                else:
                    print(f"Unknown callback name skipped: {cb}")
            elif cb.endswith('lr_monitor'):
                if self.wandb_logger is not False:
                    callback_ls.append(LearningRateMonitor(logging_interval='step'))
                else:
                    print("Wandb logger is not initialized, skipping lr_monitor callback.")
            else:
                print(f"Unknown callback name skipped: {cb}")
        
        if len(callback_ls) == 0:
            callback_ls = None
        
        trainer = Trainer(
            logger=self.wandb_logger,
            default_root_dir=self.dir_handler.output_dir,
            callbacks=callback_ls,
            max_epochs=self.train_config.max_epochs,
        )
        return trainer
                
    
    @final
    def fit(self):
        self.model_config['mode'] = 'train'
        self.data_config['mode'] = 'train'
        self.train_config['mode'] = 'train'
        
        # trainer initialization
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.dir_handler.output_dir,
            filename='{epoch}-{val_loss:.4f}', 
            monitor='val_loss',
            mode='min',
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            max_epochs=self.train_config.max_epochs,
            logger=self.wandb_logger,
            callbacks=[lr_monitor, checkpoint_callback] if self.wandb_logger is not False else None,
            default_root_dir=self.dir_handler.output_dir,
        )
        precision = getattr(self.train_config, 'matmul_precision', 'highest')
        torch.set_float32_matmul_precision(precision)
        trainer.fit(self.pipeline, datamodule=self.datamodule)
    
    
    def debug_fit(self):
        self.model_config['mode'] = 'debug'
        self.data_config['mode'] = 'debug'
        self.train_config['mode'] = 'debug'
        trainer = Trainer(
            max_epochs=self.train_config.max_epochs,
            logger=self.wandb_logger,
            default_root_dir=self.dir_handler.output_dir,
        )
        precision = getattr(self.train_config, 'matmul_precision', 'highest')
        torch.set_float32_matmul_precision(precision)
        trainer.fit(self.pipeline, datamodule=self.datamodule)
    
    @final
    def probe_fit(self):
        self.model_config['mode'] = 'probe'
        self.data_config['mode'] = 'probe'
        self.train_config['mode'] = 'probe'
        self.setup_probe_pipeline()
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.dir_handler.output_dir,
            filename='{epoch}-{probe_val_loss:.4f}', 
            monitor='probe_val_loss',
            mode='min',
            save_last=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            max_epochs=self.probe_config.max_epochs,
            logger=self.wandb_logger,
            callbacks=[lr_monitor, checkpoint_callback],
            default_root_dir=self.dir_handler.output_dir,
        )
        trainer.fit(self.probe_pipeline, datamodule=self.datamodule)

    @final
    def probe_test(self, pos_label):
        self.data_config['mode'] = 'probe_test'
        self.model_config['mode'] = 'probe_test'
        self.train_config['mode'] = 'probe_test'
        
        trainer = Trainer(
            max_epochs=self.probe_config.max_epochs,
            logger=self.wandb_logger,
            # callbacks=[lr_monitor, checkpoint_callback],
            default_root_dir=self.dir_handler.output_dir,
        )
        trainer.test(self.probe_pipeline, datamodule=self.datamodule)
        return self.probe_pipeline.process_and_reset_channel_loss(pos_label)
    
    @final
    def test(self):
        self.model_config['mode'] = 'test'
        self.data_config['mode'] = 'test'
        self.train_config['mode'] = 'test'
        
        trainer = Trainer(
            max_epochs=self.train_config.max_epochs,
            logger=self.wandb_logger,
            # callbacks=[lr_monitor, checkpoint_callback],
            default_root_dir=self.dir_handler.output_dir,
        )
        trainer.test(self.pipeline, datamodule=self.datamodule)
    
    ## ----------------- Cumstomized functions ----------------- ##
    def get_training_name(self):
        """
        Returns:
            str: A unique name for the training run.
        
        To override this function, you can create a subclass of TrainingManagerBase and redefine the get_training_name method.
        For example:

        ```python
        class CustomTrainingManager(TrainingManagerBase):
            def get_training_name(self):
                # Custom logic for generating the training name
                custom_name = f'CustomRun_L{self.model_config.num_layers}_H{self.model_config.num_heads}_' + time.strftime("%Y%m%d-%H%M%S")
                print(f"Custom training run: {custom_name}")
                return custom_name
        ```
        Then, you can use CustomTrainingManager instead of TrainingManagerBase when initializing your training manager.
        """
        training_name = time.strftime("%m%d-%H%M%S")
        print(f"Current training run: {training_name}")
        group_info = None
        return training_name, group_info

    def config_datamodule(self):
        """
        Generates a dictionary of keyword arguments for the data module.
        To override this function, you should keep in mind that the returned dictionary should respect the lightening data module's __init__ method signature.
        
        Returns:
            dict: A dictionary with 'data_config' and 'dir_handler' as keys.
        """
        # check if the batch_size keyword is in the data_config
        if "batch_size" not in self.data_config.to_dict().keys():
            self.data_config.batch_size = self.train_config.batch_size
        return {
            "data_config": self.data_config, 
            "dir_handler": self.dir_handler,
        }

    def config_pipeline(self):
        """
        Generates a dictionary of keyword arguments required for setting up the training pipeline.
        
        Returns:
            dict: A dictionary containing the arguments required for initializing the pipeline.
        """
        training_model = GPT2Standard(self.model_config)
        if self.train_config.loss_type == "cross_entropy":
            loss_p_model = nn.CrossEntropyLoss()
            loss_n_model = None
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model
        }
    
    def config_probepipeline(self):
        raise NotImplementedError("This function should be implemented in the subclass.")