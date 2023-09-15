/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.tflite_classification

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.ActivityResultRegistry
import androidx.activity.result.contract.ActivityResultContracts
import com.example.tflite_classification.databinding.ActivityMainBinding
import androidx.activity.result.contract.ActivityResultContracts.GetContent
import androidx.core.app.ActivityCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.example.tflite_classification.callbacks.ImageCallback
import com.example.tflite_classification.ml.FinalModel
import kotlinx.coroutines.currentCoroutineContext
import org.tensorflow.lite.support.image.TensorImage

// 라이프사이클이 소멸되면 런처도 삭제되기 때문에, 리소스 관리하기 좋다.
class MyLifecycleObserver(
        private val registry: ActivityResultRegistry,
        private val imageCaptureCallback: ImageCallback
    ): DefaultLifecycleObserver {

        lateinit var getContent: ActivityResultLauncher<String>
        lateinit var takePicture: ActivityResultLauncher<Void?>
        var pickedBitmap: Bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)

    override fun onCreate(owner: LifecycleOwner) {
        super.onCreate(owner)

        // Get image from gallery
        getContent = registry.register("key", owner, GetContent()) { uri ->
            Log.d("uriuir", uri.toString())
            imageCaptureCallback.onUriReceived(uri?: Uri.EMPTY)
        }

        // Get image from camera
        takePicture = registry.register(
            "cameraKey", owner, ActivityResultContracts.TakePicturePreview()) {
            Log.i("Bitmap?", it.toString())
            imageCaptureCallback.onImageReceived(it?:Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888))
        }
    }

    fun selectImage() {
        getContent.launch("image/*")
    }

    fun startCamera() {
        takePicture.launch(null)
    }

    } // end lifecycleObserver


class MainActivity : AppCompatActivity(), ImageCallback {
    private lateinit var activityMainBinding: ActivityMainBinding

    private lateinit var cameraButton: Button
    lateinit var galleryButton: Button
    lateinit var mainImageView: ImageView
    private lateinit var mainTextView: TextView

    // lifecycle register
    lateinit var observer: MyLifecycleObserver

    // 외부 액티비티 실행 결과 처리
    private val getContent = registerForActivityResult(GetContent()) {
        Log.d("returnedUri", it.toString());
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        cameraButton = activityMainBinding.button
        galleryButton = activityMainBinding.button2

        mainTextView = activityMainBinding.result
        mainImageView = activityMainBinding.imageView

        // 라이프사이클 레지스터 등록
        observer = MyLifecycleObserver(this.activityResultRegistry, this)
        lifecycle.addObserver(observer)

        cameraButton.setOnClickListener{
            if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                observer.startCamera()
                mainImageView.setImageBitmap(observer.pickedBitmap)
                Log.d("eun", observer.pickedBitmap.toString())
            } else {
                // 101
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), 101)
            }
        }

        galleryButton.setOnClickListener {
            observer.selectImage()
            Log.d("imageok", "image set ok")
        }
    }

    override fun onBackPressed() {
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            // Workaround for Android Q memory leak issue in IRequestFinishCallback$Stub.
            // (https://issuetracker.google.com/issues/139738913)
            finishAfterTransition()
        } else {
            super.onBackPressed()
        }
    }

    override fun onImageReceived(bitmap: Bitmap) {
        mainImageView.setImageBitmap(bitmap)

//----------------------tflite classification---------------------------------
        val model = FinalModel.newInstance(this)

        // Creates inputs for reference.
        val image = TensorImage.fromBitmap(bitmap)

        // Runs model inference and gets result.
        val outputs = model.process(image)
        val probability = outputs.probabilityAsCategoryList

        val sortedProb = probability.sortedByDescending { it.score }
        val highestLabel = sortedProb[0].label

        mainTextView.text = highestLabel
        if (mainTextView.text == "empty") {
            mainTextView.setTextColor(Color.RED)
        } else {
            mainTextView.setTextColor(Color.GREEN)
        }

        // Releases model resources if no longer used.
        model.close()
        //------------------------------------------------------------------------------

    }

    override fun onUriReceived(uri: Uri) {
        val resolver = this.contentResolver
        resolver.openInputStream(uri)?.use {inputStream ->
            val bitmap = BitmapFactory.decodeStream(inputStream)
            mainImageView.setImageBitmap(bitmap)
            Log.d("eun", "onUriReceived...")

            //----------------------tflite classification---------------------------------
            val model = FinalModel.newInstance(this)

            // Creates inputs for reference.
            val image = TensorImage.fromBitmap(bitmap)

            // Runs model inference and gets result.
            val outputs = model.process(image)
            val probability = outputs.probabilityAsCategoryList

            val sortedProb = probability.sortedByDescending { it.score }
            val highestLabel = sortedProb[0].label

            mainTextView.text = highestLabel
            if (mainTextView.text == "empty") {
                mainTextView.setTextColor(Color.RED)
            } else {
                mainTextView.setTextColor(Color.GREEN)
            }

            // Releases model resources if no longer used.
            model.close()
            //------------------------------------------------------------------------------
        }
    }
}