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

    <h1>Appliance Schedule</h1>
    @foreach ($schedule as $day => $daySchedule)
        <h2>{{ $day }}</h2>
        <table border="1">
            <thead>
                <tr>
                    <th>Hour</th>
                    <th>Appliance</th>
                    <th>Power (kW)</th>
                    <th>Price (€/MWh)</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($daySchedule as $hour => $entry)
                    <tr>
                        <td>{{ $hour }}:00</td>
                        <td>{{ $entry['appliance'] }}</td>
                        <td>{{ $entry['power'] }}</td>
                        <td>{{ $entry['price'] }}</td>
                    </tr>
                @endforeach
            </tbody>
        </table>
    @endforeach
</body>

</html>