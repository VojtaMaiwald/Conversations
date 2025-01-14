package eu.siacs.conversations.emotion_recognition;

import android.Manifest;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Switch;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.android.odml.image.BitmapMlImageBuilder;
import com.google.android.odml.image.MlImage;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;

import eu.siacs.conversations.Config;
import eu.siacs.conversations.entities.Conversation;
import eu.siacs.conversations.ui.ConversationFragment;
import eu.siacs.conversations.ui.ConversationsActivity;

public class FaceDetectorThread extends Thread {
    private final String CLASSIFIER = "MobileNetV2.tflite";
    private final int emotions = 8;

    private final ConversationFragment fragment;
    private final ConversationsActivity activity;
    ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;
    private MlImage mlImage;
    private Bitmap bitmap;
    private Bitmap faceBitmap;
    private FaceDetector detector;
    private Face face;
    private Interpreter interpreter;
    private int width;
    private int height;
    private EmotionList<float[]> emotionList;
    private boolean stopped = true;

    public FaceDetectorThread(ConversationsActivity activity, ConversationFragment fragment) {
        this.activity = activity;
        init();
        setupCamera();
        setupFaceDetector();
        //setupEmotionDetector();
        this.stopped = false;
        this.fragment = fragment;
    }

    public boolean isStopped() {
        return this.stopped;
    }

    @Override
    public void run() {
        Log.d(Config.LOGTAG, "FaceDetectorThread.run()");
        Log.wtf("emotionsDetections", "FaceDetectorThread.run() - detection started");
        setupEmotionDetector();
        try {
            sleep(2000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        int counter = 0;
        boolean faceAvailable = true;
        while (!stopped) {
            if (imageCapture != null) {
                takePicture();
                detectFaces();
                detectEmotions();
                if (faceAvailable) {
                    counter ++;
                }
            }

            try {
                sleep(300);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            if (counter >= 5 && faceBitmap != null && face != null && fragment.getConversation().getContact().isActive()) {
                fragment.sendEmotionMessage(fragment.getConversation().getAccount(), fragment.getConversation().getContact(), "" + emotionList.getIndexOfMax());
                counter = 0;
                faceAvailable = true;
            }
            if (counter > 5 && faceAvailable) {
                faceAvailable = false;
                fragment.sendEmotionMessage(fragment.getConversation().getAccount(), fragment.getConversation().getContact(), "8");
            }
        }
        Log.d(Config.LOGTAG, "FaceDetectorThread.run() - detection stopped");
        Log.wtf("emotionsDetections", "FaceDetectorThread.run() - detection stopped");
        fragment.sendEmotionMessage(fragment.getConversation().getAccount(), fragment.getConversation().getContact(), "-1");
    }

    public void stopDetection() {
        stopped = true;
    }

    private void init() {
        emotionList = new EmotionList<>(10, emotions);
        ActivityCompat.requestPermissions(this.activity, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ActivityCompat.requestPermissions(this.activity, new String[]{Manifest.permission.MANAGE_EXTERNAL_STORAGE}, 2);
        }
    }

    private void setupCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this.activity.getApplicationContext());
        try {
            cameraProvider = cameraProviderFuture.get();
            Preview preview = new Preview.Builder().build();
            CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
            imageCapture = new ImageCapture.Builder().setTargetRotation(Surface.ROTATION_0).setTargetResolution(new Size(480, 640)).build();
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this.activity, cameraSelector, imageCapture, preview);
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void detectFaces() {
        if (mlImage == null) {
            return;
        }
        detector.process(mlImage).addOnSuccessListener(faces -> {
            if (faces.isEmpty()) {
                this.face = null;
            } else {
                this.face = faces.get(0);
                refreshFaceBitmap(this.face.getBoundingBox());
            }
        }).addOnFailureListener(e -> {
        });
    }

    private void takePicture() {
        imageCapture.takePicture(ContextCompat.getMainExecutor(this.activity.getApplicationContext()), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                buffer.rewind();
                byte[] bytes = new byte[buffer.capacity()];
                buffer.get(bytes);
                byte[] clonedBytes = bytes.clone();
                bitmap = BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length);
                Matrix matrix = new Matrix();
                matrix.postRotate(-90);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                mlImage = new BitmapMlImageBuilder(bitmap).build();
                image.close();
            }
        });
    }

    private void setupFaceDetector() {
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setMinFaceSize(0.15f)
                .enableTracking()
                .build();

        detector = FaceDetection.getClient(faceDetectorOptions);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.activity.getAssets().openFd(CLASSIFIER);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.getStartOffset(), assetFileDescriptor.getDeclaredLength());
    }

    private void setupEmotionDetector() {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            delegateOptions.setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            interpreterOptions.addDelegate(gpuDelegate);
            Log.wtf("emotions", "GPU delegate supported");
        } else {
            Log.wtf("emotions", "GPU delegate not supported");
        }
        try {
            if (interpreter != null) {
                interpreter.close();
            }
            interpreter = new Interpreter(loadModelFile(), interpreterOptions);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void detectEmotions() {
        if (faceBitmap == null) {
            emotionList.removeLast();
            return;
        }
        DataType inputDataType = interpreter.getInputTensor(0).dataType();
        TensorImage tensorImage = new TensorImage(inputDataType);
        tensorImage.load(faceBitmap);
        FloatBuffer output = FloatBuffer.allocate(emotions);

        try {
            long startTime = System.currentTimeMillis();
            interpreter.run(tensorImage.getBuffer(), output);
            long difference = System.currentTimeMillis() - startTime;
            Log.wtf("emotionsDetectionsTime", difference + " ms");
            emotionList.add(output.array());
        }
        catch (Exception ignored) {

        }

        Log.wtf("emotionsDetections", Arrays.toString(emotionList.getTail()));
    }

    private void refreshFaceBitmap(Rect rect) {
        if (bitmap == null || rect.left < 0 || rect.top < 0 || rect.width() + Math.max(0, rect.left) > bitmap.getWidth() || rect.height() + Math.max(0, rect.top) > bitmap.getHeight()) {
            return;
        }
        faceBitmap = Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height()), 224, 224, true);
    }
}
