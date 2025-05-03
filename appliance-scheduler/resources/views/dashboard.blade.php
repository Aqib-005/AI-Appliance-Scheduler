<!DOCTYPE html>
<html>

<head>
    <title>Dashboard</title>
    <link href="{{ asset('css/app.css') }}" rel="stylesheet">
</head>

<body class="dashboard-page">
    <header class="app-header">
        <div class="app-header-container">
            <a href="{{ route('dashboard') }}" class="app-header-brand">
                <img src="{{ asset('images/logo.png') }}" alt="App Logo" class="app-header-logo">
                <span class="app-header-title">HomeSched</span>
            </a>
            <h1 class="app-page-title">Dashboard</h1>
        </div>
    </header>

    <div class="container">
        <div class="left">
            <div class="button-container">
                <h2>Scheduled Appliances</h2>
                <a href="{{ route('schedule.create') }}"><button>Schedule</button></a>
                <button id="toggleFullscreenBtn">Fullscreen</button>
            </div>

            <!-- Display total weekly cost -->
            <div class="weekly-cost">
                Weekly Cost: €{{ number_format($weeklyCost, 2) }}
            </div>

            <!-- Timetable display -->
            <div class="timetable-wrapper" id="timetableContainer">
                <div class="timetable-grid">
                    <div class="title">Time</div>
                    <!-- Generate weekday labels -->
                    @php
                        $dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
                    @endphp
                    @foreach($dayNames as $day)
                        <div class="title">{{ $day }}</div>
                    @endforeach

                    <!-- Time labels and empty grid cells -->
                    @for($h = 0; $h < 24; $h++)
                        <div class="time-label">{{ sprintf('%02d:00', $h) }}</div>
                        @for($d = 0; $d < 7; $d++)
                            <div></div>
                        @endfor
                    @endfor
                </div>

                @php
                    // Group schedule entries by day
                    $byDay = [];
                    foreach ($schedule as $e) {
                        $byDay[$e->day][] = $e;
                    }
                    // Sort each day by start time
                    foreach ($byDay as &$list) {
                        usort($list, fn($a, $b) => $a->start_hour <=> $b->start_hour);
                    }
                @endphp

                <div class="blocks-container">
                    @foreach($byDay as $day => $entries)
                        @php $d = array_search($day, $dayNames); @endphp
                        @foreach($entries as $entry)
                            @php
                                $s = $entry->start_hour;
                                $e = $entry->end_hour;

                                // Handle overlapping blocks
                                $overlaps = array_filter(
                                    $entries,
                                    fn($x) => $x->start_hour < $e && $x->end_hour > $s
                                );
                                usort($overlaps, fn($a, $b) => $a->start_hour <=> $b->start_hour);
                                $count = count($overlaps);
                                $pos = array_search($entry, $overlaps, true);

                                // Assign color and positioning
                                $hue = crc32($entry->appliance_id . $s . $day) % 360;
                                $bg = "hsla($hue,70%,50%,0.4)";
                                $border = "hsla($hue,70%,40%,1)";
                                $top = "calc(var(--row-height) * (1 + $s))";
                                $height = "calc(var(--row-height) * (" . ($e - $s) . "))";
                                $width = "calc(var(--col-width) / $count)";
                                $left = "calc(var(--first-col) + var(--col-width) * $d + ($width) * $pos)";
                            @endphp

                            <!-- Render a colored block for this appliance schedule -->
                            <div class="appliance-block" style="
                                                                            top: {{ $top }};
                                                                            height: {{ $height }};
                                                                            left: {{ $left }};
                                                                            width: {{ $width }};
                                                                            background-color: {{ $bg }};
                                                                            border: 1px solid {{ $border }};
                                                                        ">
                                {{ $entry->appliance->name }}
                            </div>
                        @endforeach
                    @endforeach
                </div>
            </div>
        </div>

        <div class="right">
            <!-- Appliance list -->
            <div class="window">
                <h2>Appliances</h2>
                <ul>
                    @foreach($appliances as $a)
                        <li>{{ $a->name }} ({{ $a->power }} kW/h)</li>
                    @endforeach
                </ul>
                <a href="{{ route('appliances.manage') }}"><button>View All Appliances</button></a>
            </div>

            <!-- Electricity price predictions chart -->
            <div class="window">
                <h2>Predicted Prices</h2>
                @if(!empty($predictions))
                    @php
                        // Format prediction data for Chart.js
                        $chartData = collect($predictions)->map(fn($p) => [
                            'x' => \Carbon\Carbon::parse($p['StartDateTime'])->toDateTimeString(),
                            'y' => $p['Predicted_Price']
                        ]);
                    @endphp
                    <canvas id="predictionsChart"></canvas>

                    <!-- Chart.js and time adapter -->
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/moment@2.29.3/moment.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment"></script>

                    <!-- Render line chart -->
                    <script>
                        const ctx = document.getElementById('predictionsChart').getContext('2d');
                        new Chart(ctx, {
                            type: 'line',
                            data: {
                                datasets: [{
                                    label: 'Price (€/MWh)',
                                    data: {!! $chartData->toJson() !!},
                                    fill: false,
                                    tension: 0.1,
                                    borderColor: '#007bff'
                                }]
                            },
                            options: {
                                scales: {
                                    x: { type: 'time', time: { unit: 'day' } },
                                    y: { beginAtZero: false }
                                }
                            }
                        });
                    </script>
                @else
                    <p>No predictions available.</p>
                @endif
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Scroll timetable to current hour
            const now = new Date().getHours();
            const rowH = parseInt(getComputedStyle(document.documentElement)
                .getPropertyValue('--row-height'));
            const container = document.getElementById('timetableContainer');
            container.scrollTop = rowH * now - container.clientHeight / 2;

            // Fullscreen toggle logic
            const toggle = document.getElementById('toggleFullscreenBtn');
            const exitBtn = document.createElement('button');
            exitBtn.textContent = 'Exit Fullscreen';
            Object.assign(exitBtn.style, {
                position: 'absolute', top: '10px', right: '10px',
                background: '#dc3545', color: '#fff', border: 'none',
                padding: '6px 12px', cursor: 'pointer', display: 'none', zIndex: 10
            });
            container.appendChild(exitBtn);

            toggle.onclick = () => {
                Object.assign(container.style, {
                    position: 'fixed', top: 0, left: 0,
                    width: '100vw', height: '100vh'
                });
                toggle.style.display = 'none';
                exitBtn.style.display = 'block';
            };

            exitBtn.onclick = () => {
                Object.assign(container.style, {
                    position: '', top: '', left: '',
                    width: '', height: ''
                });
                toggle.style.display = '';
                exitBtn.style.display = 'none';
            };
        });
    </script>
</body>

</html>