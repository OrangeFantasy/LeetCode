using Microsoft.Win32;
using System;
using System.Threading.Tasks;
using YoloGui.Commands;
using YoloGui.Main;

namespace YoloGui.ViewModels
{
    class MainWindowViewModel : NotificationObject
    {
        public MainWindowViewModel()
        {
            LoadEigenCommand = new DelegateCommand
            {
                ExecuteAction = new Action<object>(LoadEngine)
            };
            LoadLabelsCommand = new DelegateCommand
            {
                ExecuteAction = new Action<object>(LoadLabels)
            };
            LoadInputCommand = new DelegateCommand
            {
                ExecuteAction = new Action<object>(LoadInput)
            };
            StartInferenceCommand = new DelegateCommand
            {
                ExecuteAction = new Action<object>(StartInference)
            };
            StopInferenceCommand = new DelegateCommand
            {
                ExecuteAction = new Action<object>(StopInference)
            };
        }

        private string enginePath = "EnginePath";
        public string EnginePath
        {
            get => enginePath;
            set { enginePath = value; RaisePropertyChange("EnginePath"); }
        }

        private string labelsPath = "Labels Path";
        public string LabelsPath
        {
            get => labelsPath;
            set { labelsPath = value; RaisePropertyChange("LabelsPath"); }
        }

        private string inputPath = "Input Path";
        public string InputPath
        {
            get => inputPath;
            set { inputPath = value; RaisePropertyChange("InputPath"); }
        }

        private string inputType = "Camera";
        public string InputType
        {
            get => inputType;
            set { inputType = value; RaisePropertyChange("InputType"); }
        }

        private bool modelNotRunning = true;
        public bool ModelNotRunning
        {
            get => modelNotRunning;
            set { modelNotRunning = value; RaisePropertyChange("ModelNotRunning"); }
        }

        public void LoadEngine(object parameter)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Select Eigen File";
            openFileDialog.DefaultExt = ".engine";
            openFileDialog.Filter = "Engine files (*.engine)|*.engine|All files (*.*)|*.*";

            if (openFileDialog.ShowDialog() == true)
            {
                EnginePath = openFileDialog.FileName;
            }
        }

        public void LoadLabels(object parameter)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Select Labels File";
            openFileDialog.DefaultExt = ".txt";
            openFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";

            if (openFileDialog.ShowDialog() == true)
            {
                LabelsPath = openFileDialog.FileName;
            }
        }

        public void LoadInput(object parameter)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Select Input File";
            openFileDialog.DefaultExt = ".mp4";
            openFileDialog.Filter = "Madia files (*.mp4)|*.mp4|";

            if (openFileDialog.ShowDialog() == true)
            {
                LabelsPath = openFileDialog.FileName;
            }
        }

        public void StartInference(object parameter)
        {
            Task.Run(() =>
            {
                InferenceThread();
                return Task.CompletedTask;
            });

            ModelNotRunning = false;
        }

        public void StopInference(object parameter)
        {
            YoloProxy.StropRunModel();
        }

        public void InferenceThread()  // Run on the InferenceThread.
        {
            YoloProxy.ModelInputType modelInputType = new YoloProxy.ModelInputType();
            switch (InputType)
            {
                case "Cameara":
                    modelInputType = YoloProxy.ModelInputType.Camera; break;
                case "Video":
                    modelInputType = YoloProxy.ModelInputType.Video; break;
                case "Image":
                    modelInputType = YoloProxy.ModelInputType.Image; break;
                default:
                    break;
            }

            YoloProxy.LoadLabels(LabelsPath);
            YoloProxy.RunModel(EnginePath, InputPath, modelInputType);
        }

        public DelegateCommand LoadEigenCommand { get; set; }
        public DelegateCommand LoadLabelsCommand { get; set; }
        public DelegateCommand LoadInputCommand { get; set; }
        public DelegateCommand StartInferenceCommand { get; set; }
        public DelegateCommand StopInferenceCommand { get; set; }
    }
}
