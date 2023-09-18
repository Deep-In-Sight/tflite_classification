package com.example.tflite_classification.ml;

import android.content.Context;
import androidx.annotation.NonNull;
import java.io.IOException;
import java.lang.Integer;
import java.lang.Object;
import java.lang.String;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * Identify the most prominent object in the image from a set of 2 categories. */
public final class FinalModel {
  @NonNull
  private final ImageProcessor imageProcessor;

  private int imageHeight;

  private int imageWidth;

  @NonNull
  private final List<String> labels;

  @NonNull
  private final TensorProcessor probabilityPostProcessor;

  @NonNull
  private final Model model;

  private FinalModel(@NonNull Context context, @NonNull Model.Options options) throws IOException {
    model = Model.createModel(context, "final_model.tflite", options);
    MetadataExtractor extractor = new MetadataExtractor(model.getData());
    ImageProcessor.Builder imageProcessorBuilder = new ImageProcessor.Builder()
      .add(new ResizeOp(100, 100, ResizeMethod.NEAREST_NEIGHBOR))
      .add(new NormalizeOp(new float[] {0.0f}, new float[] {255.0f}))
      .add(new QuantizeOp(0f, 0.0f))
      .add(new CastOp(DataType.FLOAT32));
    imageProcessor = imageProcessorBuilder.build();
    TensorProcessor.Builder probabilityPostProcessorBuilder = new TensorProcessor.Builder()
      .add(new DequantizeOp((float)0, (float)0.0))
      .add(new NormalizeOp(new float[] {0.0f}, new float[] {1.0f}));
    probabilityPostProcessor = probabilityPostProcessorBuilder.build();
    labels = FileUtil.loadLabels(extractor.getAssociatedFile("labels.txt"));
  }

  @NonNull
  public static FinalModel newInstance(@NonNull Context context) throws IOException {
    return new FinalModel(context, (new Model.Options.Builder()).build());
  }

  @NonNull
  public static FinalModel newInstance(@NonNull Context context, @NonNull Model.Options options)
      throws IOException {
    return new FinalModel(context, options);
  }

  @NonNull
  public Outputs process(@NonNull TensorImage image) {
    imageHeight = image.getHeight();
    imageWidth = image.getWidth();
    TensorImage processedimage = imageProcessor.process(image);
    Outputs outputs = new Outputs(model);
    model.run(new Object[] {processedimage.getBuffer()}, outputs.getBuffer());
    return outputs;
  }

  public void close() {
    model.close();
  }

  @NonNull
  public Outputs process(@NonNull TensorBuffer image) {
    TensorBuffer processedimage = image;
    Outputs outputs = new Outputs(model);
    model.run(new Object[] {processedimage.getBuffer()}, outputs.getBuffer());
    return outputs;
  }

  public class Outputs {
    private TensorBuffer probability;

    private Outputs(Model model) {
      this.probability = TensorBuffer.createFixedSize(model.getOutputTensorShape(0), DataType.FLOAT32);
    }

    @NonNull
    public List<Category> getProbabilityAsCategoryList() {
      return new TensorLabel(labels, probabilityPostProcessor.process(probability)).getCategoryList();
    }

    @NonNull
    public TensorBuffer getProbabilityAsTensorBuffer() {
      return probabilityPostProcessor.process(probability);
    }

    @NonNull
    private Map<Integer, Object> getBuffer() {
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, probability.getBuffer());
      return outputs;
    }
  }
}
