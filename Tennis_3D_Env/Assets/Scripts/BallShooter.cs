using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallShooter : MonoBehaviour
{

    public GameObject prefab;
    Vector3 shooterPosition;
    public float shootForce;
    public float upwardForce;
    public float sideForce;
    // Start is called before the first frame update
    void Start()
    {
        // prefab = Resources.Load("tennisball") as GameObject;
        shooterPosition = transform.position;
        shooterPosition.z = shooterPosition.z - 0.3f;
    }

    // Update is called once per frame
    void Update()
    {   
        if(Input.GetMouseButtonDown(0)){
            GameObject tennisball = Instantiate(prefab) as GameObject;
            
            tennisball.transform.position = shooterPosition;
            Rigidbody rb = tennisball.GetComponent<Rigidbody>();

            Vector3 dir = Camera.main.transform.position - shooterPosition;

            rb.velocity = dir.normalized * shootForce + new Vector3(sideForce, upwardForce, 0);

        }
    }
}
