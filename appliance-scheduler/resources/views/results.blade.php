<!DOCTYPE html>
<html>

<head>
    <title>Prediction Results</title>
</head>

<body>
    <h1>Predicted Electricity Prices</h1>
    <table border="1">
        <thead>
            <tr>
                <th>Start Date/Time</th>
                <th>Predicted Price (€/MWh)</th>
            </tr>
        </thead>
        <tbody>
            @foreach ($predictions as $prediction)
                <tr>
                    <td>{{ $prediction['Start date/time'] }}</td>
                    <td>{{ $prediction['Predicted Price [Euro/MWh]'] }}</td>
                </tr>
            @endforeach
        </tbody>
    </table>
</body>

</html>