package com.example.surp2023;


import static android.content.ContentValues.TAG;
import static android.os.Environment.DIRECTORY_DOWNLOADS;

import static java.lang.String.valueOf;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.DownloadManager;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.IntentSender;
import android.content.pm.PackageManager;

import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;

import android.location.LocationManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.*;

import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.common.api.ResolvableApiException;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.LocationSettingsRequest;
import com.google.android.gms.location.LocationSettingsResponse;
import com.google.android.gms.location.LocationSettingsStatusCodes;
import com.google.android.gms.location.Priority;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.Timestamp;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.FieldValue;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
//import com.google.firebase.firestore.DocumentReference;
//import com.google.firebase.firestore.FirebaseFirestore;

import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.SensorEvent;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {

    private static int MICROPHONE_PERMISSION_CODE = 200;
    // denotes the minimum interval at which to gather data (in milliseconds)
    final int intervalMillis = 1000;
    // denotes the fastest possible interval at which to gather data (in milliseconds)
    final int minUpdateIntervalMillis = 500;
    private TextView CoordinateText;
    private Button LocationButton;
    private LocationRequest locationRequest;
    private LocationRequest.Builder builder;

    private TextView xRotationText;
    private TextView yRotationText;
    private TextView zRotationText;

    private TextView timestampText;

    MediaRecorder mediaRecorder;
    MediaPlayer mediaPlayer;

    ExecutorService executorService = Executors.newSingleThreadExecutor();

    // private FirebaseFirestore db = FirebaseFirestore.getInstance();
    //private DocumentReference mDocRef = db.document("users");


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        builder = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, intervalMillis);
        builder.setMinUpdateIntervalMillis(minUpdateIntervalMillis);

        CoordinateText = findViewById(R.id.addressText);
        LocationButton = findViewById(R.id.locationButton);

        xRotationText = findViewById(R.id.xRotation);
        yRotationText = findViewById(R.id.yRotation);
        zRotationText = findViewById(R.id.zRotation);
        timestampText = findViewById(R.id.timestamp);

        locationRequest = builder.build();

        final int[] count = {0};

        // Code to get audio mic permission
        if(isMicPresent()){
            getMicPermission();
        }


        // Making something happen when the button is pressed
        LocationButton.setOnClickListener(new View.OnClickListener(){
            @Override
            // Need to check for user permissions first
            public void onClick(View v){
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED){
                        if(isGPSEnabled()){
                            showSensorValues();
                            LocationServices.getFusedLocationProviderClient(MainActivity.this)
                                    .requestLocationUpdates(locationRequest, new LocationCallback() {
                                        @Override
                                        public void onLocationResult(@NonNull LocationResult locationResult) {
                                            super.onLocationResult(locationResult);

                                            // extracting latitude and longitude from fusedlocationproviderclient
                                            if(locationResult != null && locationResult.getLocations().size() > 0){
                                                int index = locationResult.getLocations().size() - 1;
                                                double latitude = locationResult.getLocations().get(index).getLatitude();
                                                double longitude = locationResult.getLocations().get(index).getLongitude();

                                                // temporarily adding in a counter so that we know the location is updating
                                                CoordinateText.setText("(" + count[0] + ")" + "Latitude: " + latitude + "\nLongitude: " + longitude);
                                                count[0] = count[0] + 1;
                                            }
                                        }
                                    }, Looper.getMainLooper());
                        }else{
                            turnOnGPS();
                        }
                    }else{
                        requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, 1);
                    }

                }
            }
        });


    }

    private void turnOnGPS() {
        LocationSettingsRequest.Builder builder = new LocationSettingsRequest.Builder()
                .addLocationRequest(locationRequest);
        builder.setAlwaysShow(true);

        Task<LocationSettingsResponse> result = LocationServices.getSettingsClient(getApplicationContext())
                .checkLocationSettings(builder.build());

        result.addOnCompleteListener(new OnCompleteListener<LocationSettingsResponse>() {
            @Override
            public void onComplete(@NonNull Task<LocationSettingsResponse> task) {

                try {
                    LocationSettingsResponse response = task.getResult(ApiException.class);
                    Toast.makeText(MainActivity.this, "GPS is already turned on", Toast.LENGTH_SHORT).show();

                } catch (ApiException e) {

                    switch (e.getStatusCode()) {
                        case LocationSettingsStatusCodes.RESOLUTION_REQUIRED:

                            try {
                                ResolvableApiException resolvableApiException = (ResolvableApiException)e;
                                resolvableApiException.startResolutionForResult(MainActivity.this,2);
                            } catch (IntentSender.SendIntentException ex) {
                                ex.printStackTrace();
                            }
                            break;

                        case LocationSettingsStatusCodes.SETTINGS_CHANGE_UNAVAILABLE:
                            //Device does not have location
                            break;
                    }
                }
            }
        });
    }
    // method returns true if the gps is turned on, or false if is not
    private boolean isGPSEnabled(){
        LocationManager locationManager = null;
        boolean isEnabled = false;

        locationManager = (LocationManager)getSystemService(Context.LOCATION_SERVICE);

        isEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
        return isEnabled;
    }

    public void showSensorValues(){

        // Must create a SensorManager instance first
        SensorManager sensorManager;

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        // To get list of every sensor on a device
        // List<Sensor> deviceSensors = sensorManager.getSensorList(Sensor.TYPE_ALL);

        // If there is a rotation vector sensor (ensuring that the phone's hardware works)
        if(sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null) {
            Sensor rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

            // enable the sensor activity, try to update every 10 ms
            sensorManager.registerListener(new SensorEventListener() {
                @Override
                public void onSensorChanged(SensorEvent sensorEvent) {
                    // good practice to check that we received the proper event
                    if (sensorEvent.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
                        final float[] mRotationMatrix = new float[16];
                        mRotationMatrix[0] = 1;
                        mRotationMatrix[4] = 1;
                        mRotationMatrix[8] = 1;
                        mRotationMatrix[12] = 1;

                        // convert rotation-vector to a 4x4 matrix
                        SensorManager.getRotationMatrixFromVector(mRotationMatrix, sensorEvent.values);

                        float[] orientationAngles = new float[3];
                        float[] ret = SensorManager.getOrientation(mRotationMatrix, orientationAngles);

                        // Computing angles in degrees
                        double zRotation = (orientationAngles[0]) * (180.0 / Math.PI);
                        double xRotation = orientationAngles[1] * (180.0 / Math.PI);
                        double yRotation = orientationAngles[2] * (180.0 / Math.PI);

                        // Displaying angles
                        xRotationText.setText("xRotation: " + xRotation);
                        yRotationText.setText("yRotation: " + yRotation);
                        zRotationText.setText("zRotation: " + zRotation);
                       // timestampText.setText("timestamp: " + FieldValue.serverTimestamp());

                        FirebaseFirestore db = FirebaseFirestore.getInstance();

                        // Create a new user with a first and last name
                        Map<String, Object> rotationValues = new HashMap<>();
                        rotationValues.put("xRotation: ", xRotation);
                        rotationValues.put("yRotation: ", yRotation);
                        rotationValues.put("zRotation", zRotation);
                        //rotationValues.put("timestamp: ", FieldValue.serverTimestamp());

                        // Add a new document with a generated ID
                        db.collection("GPStracktest4")
                                .add(rotationValues)
                                .addOnSuccessListener(new OnSuccessListener<DocumentReference>() {
                                    @Override
                                    public void onSuccess(DocumentReference documentReference) {
                                        Log.d(TAG, "DocumentSnapshot added with ID: " + documentReference.getId());
                                    }
                                })
                                .addOnFailureListener(new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        Log.w(TAG, "Error adding document", e);
                                    }
                                });
                    }
                }

                @Override
                public void onAccuracyChanged(Sensor sensor, int i) {
                    // do not need to do anything when the accuracy is changed
                }
            }, rotationVector, 10000);
        }else{
            xRotationText.setText("No rotation vector found");
            yRotationText.setText("No rotation vector found");
            zRotationText.setText("No rotation vector found");
        }

    }

    // Starts the audio recording
    public void recordAudio(View v){
        // Code for recording audio
            try{
                mediaRecorder = new MediaRecorder();
                mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                mediaRecorder.setOutputFile(getRecordingFilePath());
                mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
                mediaRecorder.prepare();
                mediaRecorder.start();

                Toast.makeText(this, "Recording has started", Toast.LENGTH_LONG).show();
            } catch (IOException e) {
                e.printStackTrace();
            }
    }

    public void stopAudio(View v){

        FirebaseStorage storage = FirebaseStorage.getInstance();
        StorageReference storageRef = storage.getReference();

        // try changing parameter of fromFile
        Uri file = Uri.fromFile(getRecordingFile());
        StorageReference filesRef = storageRef.child("audio/"+file.getLastPathSegment());
        UploadTask uploadTask = filesRef.putFile(file);


        // SUCCESS/FAILURE MESSAGES ON THE TOAST
        uploadTask.addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Toast.makeText(MainActivity.this, "Unsuccessful upload", Toast.LENGTH_LONG).show();
            }
        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                Toast.makeText(MainActivity.this, "Successful upload", Toast.LENGTH_LONG).show();
            }
        });

        // Give download link in LOGCAT
        storageRef.child("audio/testRecordingFile.mp3").getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
            @Override
            public void onSuccess(Uri uri) {
                // Got the download URL for 'users/me/profile.png'
                Log.w(TAG, uri.toString());

                String url = uri.toString();
                String fileName = valueOf(Timestamp.now().toDate());
                downloadFile(MainActivity.this, fileName, ".mp3", DIRECTORY_DOWNLOADS, url);
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception exception) {
                // Handle any errors
                Log.w(TAG, "didn't work");
            }
        });





        mediaRecorder.stop();
        mediaRecorder.release();
        mediaRecorder = null;

        Toast.makeText(this, "Recording has stopped", Toast.LENGTH_LONG).show();
    }

    public void downloadFile(Context context, String fileName, String fileExtension, String destinationDirectory, String url){
        DownloadManager downloadmanager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        Uri uri = Uri.parse(url);

        DownloadManager.Request request = new DownloadManager.Request(uri);
        request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
        request.setDestinationInExternalFilesDir(context, destinationDirectory, fileName + fileExtension);

        downloadmanager.enqueue(request);
    }
    public void playAudio(View v){

        try {
            mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(getRecordingFilePath());
            mediaPlayer.prepare();
            mediaPlayer.start();
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }


    // BELOW ARE HELPER FUNCTIONS FOR AUDIO MIC RECORDING

    // Check if the device has a microphone or not
    private boolean isMicPresent(){
        return this.getPackageManager().hasSystemFeature(PackageManager.FEATURE_MICROPHONE);
    }

    private void getMicPermission(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.RECORD_AUDIO}, MICROPHONE_PERMISSION_CODE);
        }
    }

    private File getRecordingFile(){
        ContextWrapper contextWrapper = new ContextWrapper(getApplicationContext());
        File musicDirectory = contextWrapper.getExternalFilesDir(Environment.DIRECTORY_MUSIC);
        File file = new File(musicDirectory, "testRecordingFile" + ".mp3");

        return file;
    }

    private String getRecordingFilePath(){
        return getRecordingFile().getPath();
    }
}