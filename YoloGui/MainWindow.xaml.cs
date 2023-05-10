using System;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using YoloGui.Main;
using YoloGui.ViewModels;

namespace YoloGui
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            readImageDelegate = new YoloProxy.ReadImageDelegate(UpdateResultImage);
            YoloProxy.RegisterReadImageDelegate(readImageDelegate);

            DataContext = new MainWindowViewModel();
        }

        YoloProxy.ReadImageDelegate readImageDelegate;

        public bool UpdateResultImage(IntPtr ImageData, int Height, int Width) // Run on the InferenceThread.
        {
            BitmapSource bitmapSource = BitmapSource.Create(Width, Height, 96.0, 96.0, PixelFormats.Bgr24, null, ImageData, Height * Width * 3, Width * 3);

            MemoryStream memoryStream = new MemoryStream();
            BitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
            encoder.Save(memoryStream);
            Bitmap bitmap = new Bitmap(memoryStream);
            bitmap.Save(memoryStream, bitmap.RawFormat);

            BitmapImage bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.StreamSource = memoryStream;
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.EndInit();
            bitmapImage.Freeze();

            Dispatcher.BeginInvoke(new Action(() =>
            {
                InferenceImage.Source = bitmapImage;
            }));

            return true;
        }

        private void ModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {

        }
    }
}
