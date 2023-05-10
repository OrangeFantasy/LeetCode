using System;
using System.Runtime.InteropServices;
using System.Text;

namespace YoloGui.Main
{
    public class YoloProxy
    {
        public delegate bool ReadImageDelegate(IntPtr ImageData, int Height, int Width);

        public enum ModelInputType
        {
            Camera,
            Video,
            Image,
        }

        private static unsafe byte[] EncodeNullTerminatedUTF8(string s)
        {
            Encoder enc = Encoding.UTF8.GetEncoder();
            fixed (char* c = s)
            {
                int len = enc.GetByteCount(c, s.Length, true);
                byte[] buf = new byte[len + 1];
                fixed (byte* ptr = buf)
                {
                    enc.Convert(c, s.Length, ptr, len, true, out _, out _, out var completed);
                }

                return buf;
            }
        }

        [DllImport("YoloV5.dll")]
        private static extern unsafe bool LoadLabels(byte* path);

        public static unsafe bool LoadLabels(string path)
        {
            fixed (byte* ptr = EncodeNullTerminatedUTF8(path))
            {
                return LoadLabels(ptr);
            }
        }

        [DllImport("YoloV5.dll")]
        public static extern unsafe void RegisterReadImageDelegate(ReadImageDelegate callback);

        [DllImport("YoloV5.dll")]
        private static extern unsafe bool RunModel(byte* engine_path, byte* input_path, int type);

        [DllImport("YoloV5.dll")]
        private static extern void StopRun();

        public static unsafe bool RunModel(string engine_path, string input_path, ModelInputType type)
        {
            fixed (byte* engine_ptr = EncodeNullTerminatedUTF8(engine_path), input_ptr = EncodeNullTerminatedUTF8(input_path))
            {
                return RunModel(engine_ptr, input_ptr, (int)type);
            }
        }

        public static void StropRunModel()
        {
            StopRun();
        }

    }
}
