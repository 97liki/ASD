$(document).ready(function() {
    $('#predictionForm').submit(function(event) {
        event.preventDefault(); // Prevent the default form submission behavior

        // Serialize the form data
        var formData = $(this).serialize();

        // AJAX request to the backend
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/predict', // Corrected
            data: formData,
            success: function(response) {
                // Assuming the response contains the prediction result
                $('#predictionResult').html('<strong>Prediction Result: </strong>' + response.result);
            },
            error: function(xhr, status, error) {
                // Handle errors
                $('#predictionResult').html('<strong>Error: </strong>' + error);
            }
        });
    });
});