using System.Collections;
using System.Collections.Generic;
using System.Collections;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class VideoRecording : MonoBehaviour
{   
    public int flipX;
    public int flipY;
    WebCamTexture webcamTexture;
    Renderer renderer;
    Color32[] data;

    RequestSocket client;

    IEnumerator FetchCoordinates(){
        while(true){
            Debug.Log("Sending Hello");
            client.SendFrame("Hello World");
            string message = null;
            bool gotMessage = false;
            gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
            if (gotMessage) Debug.Log("Received " + message);
            yield return null;
        }
    }

    // async Task<string> FetchCoordinates(){
        
    // }


    // Start is called before the first frame update
    void Start()
    {
        webcamTexture = new WebCamTexture();
        // webcamTexture.requestedHeight = 180;
        // webcamTexture.requestedWidth = 320;

        renderer = this.GetComponent<Renderer>();
        renderer.material.mainTexture = webcamTexture;
        webcamTexture.Play();

        data = new Color32[webcamTexture.width * webcamTexture.height];

        // ForceDotNet.Force();
        client =  new RequestSocket();
        client.Connect("tcp://localhost:6000");
        StartCoroutine(FetchCoordinates());
        // NetMQConfig.Cleanup();
    }

    // Update is called once per frame
    void Update()
    {
        webcamTexture.GetPixels32(data);
        
        // Debug.Log(webcamTexture.width);
        // Debug.Log(webcamTexture.height);
        // Debug.Log(data);
        
    }

    void OnDestroy(){
        NetMQConfig.Cleanup();
    }
}
