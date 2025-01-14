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

        .appliance-item button {
            background-color: #ff4d4d;
            color: white;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 3px;
        }

        .appliance-item button:hover {
            background-color: #cc0000;
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
                <div class="day-column" onclick="selectDay('{{ $day }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        @foreach ($selectedAppliances as $appliance)
                            @if (strtolower($appliance->usage_days) === strtolower($day))
                                <div class="appliance-item" id="appliance-{{ $appliance->id }}">
                                    <span>{{ $appliance->name }} ({{ $appliance->preferred_start }} -
                                        {{ $appliance->preferred_end }}, {{ $appliance->duration }} hrs)</span>
                                    <button
                                        onclick="removeAppliance({{ $appliance->id }}, '{{ strtolower($day) }}')">Remove</button>
                                </div>
                            @endif
                        @endforeach
                    </div>
                </div>
            @endforeach
        </div>
    </div>

    <!-- Schedule Button -->
    <button class="schedule-button" onclick="runSchedulingAlgorithm()">Schedule</button>

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
                    document.getElementById('selectedDayName').textContent = selectedDay || '';

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

        // Run the scheduling algorithm
        function runSchedulingAlgorithm() {
            console.log('Running scheduling algorithm...'); // Debugging

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
                    console.log('Response status:', response.status); // Debugging
                    if (!response.ok) {
                        // Log the response text for debugging
                        return response.text().then(text => {
                            console.error('Response text:', text); // Debugging
                            throw new Error('Network response was not ok');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Schedule generated:', data); // Debugging
                    if (data.success) {
                        // Redirect to the dashboard
                        window.location.href = data.redirect_url;
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

            // Save the selected appliance to the database
            fetch('/selected-appliance/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({
                    appliance_id: selectedApplianceId,
                    name: document.getElementById('applianceName').value,
                    preferred_start: preferredStart,
                    preferred_end: preferredEnd,
                    duration: duration,
                    usage_days: selectedDay, // Save as a string
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
                    <button onclick="removeAppliance(${data.appliance_id}, '${selectedDay.toLowerCase()}')">Remove</button>
                `;

                        // Add the appliance to the selected day's container
                        document.getElementById(`appliances-${selectedDay.toLowerCase()}`).appendChild(applianceItem);

                        // Close the popup
                        closeSchedulePopup();
                    } else {
                        alert('Failed to add appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error); // Debugging
                    alert('Failed to add appliance. Please try again.');
                });
        }

        function removeAppliance(applianceId, day) {
            if (!confirm('Are you sure you want to remove this appliance?')) {
                return;
            }

            // Send a DELETE request to remove the appliance
            fetch(`/selected-appliance/remove/${applianceId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
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
                        // Remove the appliance from the UI
                        const applianceElement = document.getElementById(`appliance-${applianceId}`);
                        if (applianceElement) {
                            applianceElement.remove();
                        }
                    } else {
                        alert('Failed to remove appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error); // Debugging
                    alert('Failed to remove appliance. Please try again.');
                });
        }
    </script>
</body>

</html>