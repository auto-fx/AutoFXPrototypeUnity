using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class YoloDetector : MonoBehaviour
{
    [Header("Modelo y Entrada")]
    public NNModel modelAsset;          // Modelo YOLOv8 en formato .onnx
    public Texture2D testImage;         // Imagen de prueba
    public int inputSize = 640;         // Tamaño de entrada (ej: 640x640)

    private Model runtimeModel;
    private IWorker worker;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);

        RunTestImage();
    }

    void RunTestImage()
    {
        if (testImage == null)
        {
            Debug.LogError("No se asignó imagen de prueba.");
            return;
        }

        Texture2D resized = ResizeTexture(testImage, inputSize, inputSize);
        var input = new Tensor(resized, 3); // RGB

        worker.Execute(input);
        Tensor output = worker.PeekOutput();

        List<Detection> detections = ParseYoloOutput(output, 0.4f);

        foreach (var det in detections)
        {
            Debug.Log($"Detectado: {det.label} | Confianza: {det.confidence:F2} | Caja: {det.boundingBox}");
        }

        input.Dispose();
        output.Dispose();
        Destroy(resized);
    }

    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(source, rt);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D result = new Texture2D(width, height, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    List<Detection> ParseYoloOutput(Tensor output, float confidenceThreshold = 0.4f)
    {
        List<Detection> results = new List<Detection>();

        int boxSize = 6; // x, y, w, h, conf, class_id
        int numDetections = output.shape.width;

        for (int i = 0; i < numDetections; i++)
        {
            int offset = i * boxSize;

            float x = output[offset];
            float y = output[offset + 1];
            float w = output[offset + 2];
            float h = output[offset + 3];
            float conf = output[offset + 4];
            int classIndex = Mathf.RoundToInt(output[offset + 5]);

            if (conf < confidenceThreshold)
                continue;

            string label = classIndex switch
            {
                0 => "car",
                1 => "car-tire",
                _ => "unknown"
            };

            Rect bbox = new Rect(x - w / 2f, y - h / 2f, w, h); // Normalizado

            results.Add(new Detection
            {
                label = label,
                confidence = conf,
                boundingBox = bbox
            });
        }

        return results;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}

public class Detection
{
    public string label;
    public float confidence;
    public Rect boundingBox;
}