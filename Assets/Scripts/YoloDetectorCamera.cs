using UnityEngine;
using Unity.Barracuda;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using System.Collections.Generic;

public class YoloDetectorCamera : MonoBehaviour
{
    [Header("Barracuda")]
    public NNModel modelAsset;
    public RenderTexture inputTexture;
    public GameObject boundingBoxPrefab;

    [Header("AR")]
    public Camera arCamera;
    public ARRaycastManager raycastManager;

    private IWorker worker;
    private const int inputSize = 640;
    private float timer = 0f;
    public float detectionInterval = 1.0f;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= detectionInterval)
        {
            RunDetection();
            timer = 0f;
        }
    }

    void RunDetection()
    {
        // Convertir RenderTexture a Texture2D
        Texture2D tex = new Texture2D(inputSize, inputSize, TextureFormat.RGB24, false);
        RenderTexture.active = inputTexture;
        tex.ReadPixels(new Rect(0, 0, inputSize, inputSize), 0, 0);
        tex.Apply();
        RenderTexture.active = null;

        Tensor tensor = new Tensor(tex, 3);
        worker.Execute(tensor);
        Tensor output = worker.PeekOutput();

        for (int i = 0; i < output.width; i += 6)
        {
            float x = output[0, i + 0];
            float y = output[0, i + 1];
            float w = output[0, i + 2];
            float h = output[0, i + 3];
            float conf = output[0, i + 4];
            int classIndex = Mathf.RoundToInt(output[0, i + 5]);

            if (conf > 0.6f && (classIndex == 0 || classIndex == 1)) // car or car-tire
            {
                Vector2 screenPos = new Vector2(x / inputSize * Screen.width, y / inputSize * Screen.height);
                List<ARRaycastHit> hits = new List<ARRaycastHit>();
                if (raycastManager.Raycast(screenPos, hits, TrackableType.Planes))
                {
                    Pose pose = hits[0].pose;
                    Instantiate(boundingBoxPrefab, pose.position, pose.rotation);
                    Debug.Log($"✅ Detectado: clase {classIndex} en {pose.position}");
                }
            }
        }

        tensor.Dispose();
        Destroy(tex);
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}