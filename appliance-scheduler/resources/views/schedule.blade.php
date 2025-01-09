<!DOCTYPE html>
<html>

<head>
    <title>Schedule Appliances</title>
    <style>
        /* General Styles */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }

        .container {
            display: flex;
            gap: 20px;
        }

        .appliance-list {
            flex: 1;
        }

        .weekly-grid {
            flex: 3;
            display: flex;
            gap: 10px;
        }

        .day-column {
            flex: 1;
            border: 1px solid #ccc;
            padding: 10px;
            cursor: pointer;
        }

        .day-column.active {
            background-color: #f0f0f0;
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

        .schedule-button {
            margin-top: 20px;
            padding: 10px 20px;
            cursor: pointer;
        }
    </style>
</head>

<body>
    <h1>Schedule Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button>Back to Dashboard</button>
    </a>

    <div class="container">
        <!-- Appliance List -->
        <div class="appliance-list">
            <h2>Appliances</h2>
            <ul>
                @foreach ($appliances as $appliance)
                    <li>
                        <button
                            onclick="openSchedulePopup('{{ $appliance->id }}', '{{ $appliance->name }}', '{{ $appliance->preferred_start }}', '{{ $appliance->preferred_end }}', '{{ $appliance->duration }}')">
                            {{ $appliance->name }}
                        </button>
                    </li>
                @endforeach
            </ul>
        </div>

        <!-- Weekly Grid Table -->
        <div class="weekly-grid">
            @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                <div class="day-column" onclick="selectDay('{{ strtolower($day) }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        <!-- Appliances will be dynamically added here -->
                    </div>
                </div>
            @endforeach
        </div>
    </div>

    <!-- Schedule Button -->
    <button class="schedule-button" onclick="runSchedulingAlgorithm()">Schedule</button>

    <!-- Schedule Popup -->
    <!-- Schedule Popup -->
    <div id="schedulePopup" class="popup">
        <h2>Schedule Appliance for <span id="selectedDayName"></span></h2>
        <form id="scheduleForm">
            <input type="hidden" id="applianceId" name="applianceId">
            <label for="applianceName">Appliance:</label>
            <input type="text" id="applianceName" name="applianceName" readonly>
            <br>
            <label for="preferredStart">Preferred Start Time:</label>
            <input type="time" id="preferredStart" name="preferredStart" required>
            <br>
            <label for="preferredEnd">Preferred End Time:</label>
            <input type="time" id="preferredEnd" name="preferredEnd" required>
            <br>
            <label for="duration">Duration (hours):</label>
            <input type="number" id="duration" name="duration" step="0.01" placeholder="e.g., 3.23" required>
            <br>
            <button type="button" onclick="addApplianceToDay()">Add</button>
            <button type="button" onclick="closeSchedulePopup()">Cancel</button>
        </form>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        let selectedDay = null;
        let selectedApplianceId = null;
        let selectedApplianceName = null;
        let selectedApplianceStart = null;
        let selectedApplianceEnd = null;
        let selectedApplianceDuration = null;

        // Validate time format (HH:mm)
        function isValidTime(time) {
            const regex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/; // Matches HH:mm format
            return regex.test(time);
        }

        // Open the schedule popup
        function openSchedulePopup(applianceId, applianceName, preferredStart, preferredEnd, duration) {
            console.log('Appliance ID:', applianceId);
            console.log('Appliance Name:', applianceName);
            console.log('Preferred Start:', preferredStart);
            console.log('Preferred End:', preferredEnd);
            console.log('Duration:', duration);

            selectedApplianceId = applianceId; // Set the selected appliance ID

            // Fetch the latest appliance data from the database
            fetch(`/appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    // Trim seconds from time values (if present)
                    const preferredStartFormatted = data.preferred_start ? data.preferred_start.substring(0, 5) : '';
                    const preferredEndFormatted = data.preferred_end ? data.preferred_end.substring(0, 5) : '';

                    // Populate the popup form with the latest data
                    document.getElementById('applianceId').value = data.id;
                    document.getElementById('applianceName').value = data.name;
                    document.getElementById('preferredStart').value = preferredStartFormatted;
                    document.getElementById('preferredEnd').value = preferredEndFormatted;
                    document.getElementById('duration').value = data.duration; // Ensure this matches the format
                    document.getElementById('selectedDayName').textContent = selectedDay ? selectedDay.charAt(0).toUpperCase() + selectedDay.slice(1) : '';

                    // Show the popup
                    document.getElementById('schedulePopup').classList.add('active');
                    document.getElementById('overlay').classList.add('active');
                })
                .catch(error => {
                    console.error('Error fetching appliance data:', error);
                    alert('Failed to fetch appliance data. Please try again.');
                });
        }

        // Close the schedule popup
        function closeSchedulePopup() {
            document.getElementById('schedulePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Select a day
        function selectDay(day) {
            selectedDay = day;
        }

        // Validate duration format (HH:mm)
        function isValidDuration(duration) {
            const regex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/; // Matches HH:mm format
            return regex.test(duration);
        }

        // Validate that the duration doesn't exceed the time range
        function isDurationValid(startTime, endTime, duration) {
            // Convert HH:mm to minutes
            const start = startTime.split(':');
            const end = endTime.split(':');
            const startMinutes = parseInt(start[0]) * 60 + parseInt(start[1]);
            const endMinutes = parseInt(end[0]) * 60 + parseInt(end[1]);

            // Calculate the total time range in minutes
            const timeRangeMinutes = endMinutes - startMinutes;

            // Convert duration (decimal hours) to minutes
            const durationMinutes = duration * 60;

            // Debugging logs
            console.log('Start Time (minutes):', startMinutes);
            console.log('End Time (minutes):', endMinutes);
            console.log('Duration (minutes):', durationMinutes);
            console.log('Time Range (minutes):', timeRangeMinutes);

            // Check if duration exceeds the time range
            return durationMinutes <= timeRangeMinutes;
        }

        // Run the scheduling algorithm
        function runSchedulingAlgorithm() {
            // Fetch predictions and generate the schedule
            fetch('/schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({}), // No need to send start_date
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Schedule generated:', data); // Debugging
                    if (data.success) {
                        // Redirect to the results page or update the UI with the schedule
                        window.location.href = '/results'; // Example: Redirect to the results page
                    } else {
                        alert('Failed to generate schedule. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error); // Debugging
                    alert('Failed to generate schedule. Please try again.');
                });
        }

        function addApplianceToDay() {
            if (!selectedApplianceId) {
                alert('No appliance selected. Please try again.');
                return;
            }

            const preferredStart = document.getElementById('preferredStart').value;
            const preferredEnd = document.getElementById('preferredEnd').value;
            const duration = parseFloat(document.getElementById('duration').value);

            // Validate inputs
            if (!selectedDay || !preferredStart || !preferredEnd || isNaN(duration)) {
                alert('Please fill in all fields and select a day.');
                return;
            }

            // Validate that the duration is a positive number
            if (duration <= 0) {
                alert('Duration must be a positive number.');
                return;
            }

            // Validate that the duration doesn't exceed the time range
            if (!isDurationValid(preferredStart, preferredEnd, duration)) {
                alert('Duration exceeds the preferred time range. Please adjust the duration.');
                return;
            }

            // Update the appliance in the database (using AJAX)
            fetch(`/appliance/update/${selectedApplianceId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({
                    preferred_start: preferredStart,
                    preferred_end: preferredEnd,
                    duration: duration, // Send duration as a decimal
                }),
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('API Response:', data); // Debugging
                    if (data.success) {
                        // Create the appliance item
                        const applianceItem = document.createElement('div');
                        applianceItem.className = 'appliance-item';
                        applianceItem.innerHTML = `
                    <span>${document.getElementById('applianceName').value} (${preferredStart} - ${preferredEnd}, ${duration} hrs)</span>
                `;

                        // Add the appliance to the selected day's container
                        document.getElementById(`appliances-${selectedDay}`).appendChild(applianceItem);

                        // Close the popup
                        closeSchedulePopup();
                    } else {
                        alert('Failed to update appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error); // Debugging
                    alert('Failed to update appliance. Please try again.');
                });
        }
    </script>
</body>

</html>