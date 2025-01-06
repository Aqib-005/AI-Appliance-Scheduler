<!DOCTYPE html>
<html>

<head>
    <title>Schedule Appliances</title>
    <style>
        /* General Styles */
        body {
            font-family: Arial, sans-serif;
        }

        .days-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .day {
            padding: 10px;
            border: 1px solid #ccc;
            cursor: pointer;
        }

        .day.active {
            background-color: #f0f0f0;
        }

        .appliances-list {
            margin-bottom: 20px;
        }

        .appliance-item {
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }

        .overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }
    </style>
</head>

<body>
    <h1>Schedule Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button>Back to Dashboard</button>
    </a>

    <!-- Days Navigation -->
    <div class="days-container">
        @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
            <div class="day" onclick="selectDay('{{ strtolower($day) }}')">{{ $day }}</div>
        @endforeach
    </div>

    <!-- Appliances List for Selected Day -->
    <div class="appliances-list">
        <h2 id="selected-day-header">Select a Day</h2>
        <div id="appliances-container">
            <!-- Appliances will be dynamically added here -->
        </div>
        <button onclick="openAddAppliancePopup()">Add Appliance</button>
    </div>

    <!-- Schedule Button -->
    <button onclick="runSchedulingAlgorithm()">Schedule</button>

    <!-- Add Appliance Popup -->
    <div id="addAppliancePopup" class="popup">
        <h2>Add Appliance</h2>
        <form id="addApplianceForm">
            <label for="appliance">Appliance:</label>
            <select id="appliance" name="appliance" required>
                <option value="">Select Appliance</option>
                @foreach ($appliances as $appliance)
                    <option value="{{ $appliance->id }}">{{ $appliance->name }}</option>
                @endforeach
            </select>
            <br>
            <label for="start_hour">Start Hour (0-23):</label>
            <input type="number" id="start_hour" name="start_hour" min="0" max="23" required>
            <br>
            <label for="end_hour">End Hour (0-23):</label>
            <input type="number" id="end_hour" name="end_hour" min="0" max="23" required>
            <br>
            <label for="duration">Duration (hours):</label>
            <input type="number" id="duration" name="duration" step="0.1" required>
            <br>
            <button type="button" onclick="addAppliance()">Add</button>
            <button type="button" onclick="closeAddAppliancePopup()">Cancel</button>
        </form>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        let selectedDay = null;
        const appliancesContainer = document.getElementById('appliances-container');
        const selectedDayHeader = document.getElementById('selected-day-header');

        // Select a day
        function selectDay(day) {
            selectedDay = day;
            selectedDayHeader.textContent = `Appliances for ${day.charAt(0).toUpperCase() + day.slice(1)}`;
            loadAppliancesForDay(day);
        }

        // Load appliances for the selected day
        function loadAppliancesForDay(day) {
            // Fetch appliances for the selected day (you can use AJAX)
            appliancesContainer.innerHTML = ''; // Clear the container
            // Example: Add dummy data
            const appliances = [
                { id: 1, name: 'Washing Machine', start_hour: 7, end_hour: 9, duration: 2 },
                { id: 2, name: 'Dishwasher', start_hour: 18, end_hour: 20, duration: 2 },
            ];
            appliances.forEach(appliance => {
                const applianceItem = document.createElement('div');
                applianceItem.className = 'appliance-item';
                applianceItem.innerHTML = `
                    <span>${appliance.name} (${appliance.start_hour}:00 - ${appliance.end_hour}:00, ${appliance.duration} hrs)</span>
                    <button onclick="editAppliance(${appliance.id})">Edit</button>
                `;
                appliancesContainer.appendChild(applianceItem);
            });
        }

        // Open the add appliance popup
        function openAddAppliancePopup() {
            if (!selectedDay) {
                alert('Please select a day first.');
                return;
            }
            document.getElementById('addAppliancePopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }

        // Close the add appliance popup
        function closeAddAppliancePopup() {
            document.getElementById('addAppliancePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Add an appliance
        function addAppliance() {
            const applianceId = document.getElementById('appliance').value;
            const startHour = document.getElementById('start_hour').value;
            const endHour = document.getElementById('end_hour').value;
            const duration = document.getElementById('duration').value;

            // Validate input
            if (!applianceId || !startHour || !endHour || !duration) {
                alert('Please fill in all fields.');
                return;
            }

            // Save the appliance (you can use AJAX to save to the database)
            const appliance = {
                id: applianceId,
                name: document.getElementById('appliance').selectedOptions[0].text,
                start_hour: startHour,
                end_hour: endHour,
                duration: duration,
            };

            // Add the appliance to the list
            const applianceItem = document.createElement('div');
            applianceItem.className = 'appliance-item';
            applianceItem.innerHTML = `
                <span>${appliance.name} (${appliance.start_hour}:00 - ${appliance.end_hour}:00, ${appliance.duration} hrs)</span>
                <button onclick="editAppliance(${appliance.id})">Edit</button>
            `;
            appliancesContainer.appendChild(applianceItem);

            // Close the popup
            closeAddAppliancePopup();
        }

        // Edit an appliance
        function editAppliance(applianceId) {
            // Fetch appliance details (you can use AJAX)
            const appliance = {
                id: applianceId,
                name: 'Washing Machine',
                start_hour: 7,
                end_hour: 9,
                duration: 2,
            };

            // Populate the popup form
            document.getElementById('appliance').value = appliance.id;
            document.getElementById('start_hour').value = appliance.start_hour;
            document.getElementById('end_hour').value = appliance.end_hour;
            document.getElementById('duration').value = appliance.duration;

            // Open the popup
            openAddAppliancePopup();
        }

        // Run the scheduling algorithm
        function runSchedulingAlgorithm() {
            // Collect all scheduled appliances
            const scheduledAppliances = [];
            document.querySelectorAll('.appliance-item').forEach(item => {
                const text = item.textContent.trim();
                const [name, time] = text.split(' (');
                const [startHour, endHour, duration] = time.replace(')', '').split(/[: -]/);
                scheduledAppliances.push({
                    name: name,
                    start_hour: parseInt(startHour),
                    end_hour: parseInt(endHour),
                    duration: parseFloat(duration),
                });
            });

            // Call the scheduling algorithm (you can use AJAX)
            alert('Scheduling algorithm will run with the following appliances: ' + JSON.stringify(scheduledAppliances));
        }
    </script>
</body>

</html>