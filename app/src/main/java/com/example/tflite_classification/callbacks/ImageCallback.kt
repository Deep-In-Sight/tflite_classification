package com.example.tflite_classification.callbacks

import android.graphics.Bitmap
import android.net.Uri

interface ImageCallback {
    fun onImageReceived(bitmap: Bitmap)
    fun onUriReceived(uri:Uri)
}