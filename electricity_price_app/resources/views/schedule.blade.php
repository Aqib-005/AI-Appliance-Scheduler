<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Appliance Scheduler</title>
</head>
<body>
    <h1>Appliance Scheduler</h1>
    <form action="/schedule" method="POST">
        @csrf
        <div id="appliances">
            <div class="appliance">
                <label for="appliance_name">Appliance Name:</label>
                <input type="text" name="appliances[0][name]" required><br><br>

                <label for="appliance_consumption">Consumption (MW/h):</label>
                <input type="number" name="appliances[0][consumption]" required><br><br>

                <label for="appliance_time">Preferred Time (e.g., 14:00-16:00):</label>
                <input type="text" name="appliances[0][preferred_time]" required><br><br>
            </div>
        </div>
        <button type="button" onclick="addAppliance()">Add Appliance</button><br><br>
        <button type="submit">Schedule Appliances</button>
    </form>

    @if(isset($schedule))
        <h2>Schedule:</h2>
        <ul>
            @foreach($schedule as $entry)
                <li>{{ $entry['appliance'] }} at {{ $entry['time'] }} (Price: {{ $entry['price'] }} Euro/MWh)</li>
            @endforeach
        </ul>
    @endif

    <script>
        let applianceCount = 1;

        function addAppliance() {
            const appliancesDiv = document.getElementById('appliances');
            const newAppliance = document.createElement('div');
            newAppliance.classList.add('appliance');
            newAppliance.innerHTML = `
                <label for="appliance_name">Appliance Name:</label>
                <input type="text" name="appliances[${applianceCount}][name]" required><br><br>

                <label for="appliance_consumption">Consumption (MW/h):</label>
                <input type="number" name="appliances[${applianceCount}][consumption]" required><br><br>

                <label for="appliance_time">Preferred Time (e.g., 14:00-16:00):</label>
                <input type="text" name="appliances[${applianceCount}][preferred_time]" required><br><br>
            `;
            appliancesDiv.appendChild(newAppliance);
            applianceCount++;
        }
    </script>
</body>
</html>