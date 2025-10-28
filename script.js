$(document).ready(function () {

  // Upload dataset
  $("#uploadForm").submit(function (e) {
    e.preventDefault();
    var formData = new FormData(this);
    $.ajax({
      url: "/upload",
      type: "POST",
      data: formData,
      contentType: false,
      processData: false,
      success: function (data) {
        $("#uploadPreview").html(data.preview);
        window.scrollTo({ top: $("#preprocess").offset().top, behavior: "smooth" });
      }
    });
  });

  // Preprocess
  $("#preprocessBtn").click(function () {
    $.post("/preprocess", {}, function (data) {
      $("#preprocessPreview").html(data.clean_preview);
      window.scrollTo({ top: $("#train-logistic").offset().top, behavior: "smooth" });
    });
  });

  // Train Logistic Regression
  $("#trainLogisticBtn").click(function () {
    $("#logisticResult").html("⏳ Training Logistic Regression... <br><small>Processing 200 samples with 100 features</small>");
    $.post("/train_logistic", {}, function (data) {
      $("#logisticResult").html(`✅ Logistic Regression Trained! Accuracy: ${data.logistic_accuracy}% (Trained on ${data.sample_size} samples)`);
      
      // Show charts
      $("#sentimentPie").attr("src", "/static/assets/sentiment_pie.png").show();
      $("#posWC").attr("src", "/static/assets/positive_wc.png").show();
      $("#neuWC").attr("src", "/static/assets/neutral_wc.png").show();
      $("#negWC").attr("src", "/static/assets/negative_wc.png").show();
      
      window.scrollTo({ top: $("#train-rnn").offset().top, behavior: "smooth" });
    }).fail(function(xhr) {
      $("#logisticResult").html("❌ Error: " + xhr.responseJSON.error);
    });
  });

  // Train RNN Model
  $("#trainRnnBtn").click(function () {
    $("#rnnResult").html("⏳ Training RNN Model... <br><small>Processing 100 samples with minimal architecture</small>");
    $.post("/train_rnn", {}, function (data) {
      $("#rnnResult").html(`✅ RNN Model Trained! Accuracy: ${data.rnn_accuracy}% (Trained on ${data.sample_size} samples)`);
      
      // Show RNN training chart
      $("#rnnTrainingChart").attr("src", "/static/assets/rnn_training_accuracy.png").show();
      
      window.scrollTo({ top: $("#compare-models").offset().top, behavior: "smooth" });
    }).fail(function(xhr) {
      $("#rnnResult").html("❌ Error: " + xhr.responseJSON.error);
    });
  });

  // Compare Models
  $("#compareBtn").click(function () {
    $("#compareResult").html("⏳ Comparing models... <br><small>Training both models on 100 samples</small>");
    $.post("/compare_models", {}, function (data) {
      $("#compareResult").html(`
        <div class="accuracy-comparison">
          <h3>📊 Model Accuracy Comparison</h3>
          <p><small>⚡ Ultra-fast training on ${data.sample_size} samples (completed in seconds!)</small></p>
          <div class="accuracy-cards">
            <div class="accuracy-card logistic">
              <h4>Logistic Regression</h4>
              <div class="accuracy-score">${data.logistic_accuracy}%</div>
            </div>
            <div class="accuracy-card rnn">
              <h4>RNN (TensorFlow)</h4>
              <div class="accuracy-score">${data.rnn_accuracy}%</div>
            </div>
          </div>
        </div>
      `);
      
      // Show comparison chart
      $("#comparisonChart").attr("src", "/static/assets/model_comparison.png").show();
      
      window.scrollTo({ top: $("#predict").offset().top, behavior: "smooth" });
    }).fail(function(xhr) {
      $("#compareResult").html("❌ Error: " + xhr.responseJSON.error);
    });
  });

  // Predict
  $("#predictForm").submit(function (e) {
    e.preventDefault();
    $.post("/predict", $(this).serialize(), function (data) {
      $("#predictResult").html("<h3>Prediction: " + data.label + "</h3>");
    });
  });

});
