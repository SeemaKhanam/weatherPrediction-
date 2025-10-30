document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found');
        return;
    }

    const ctx = chartElement.getContext('2d');

    // Corrected gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 100);
    gradient.addColorStop(0, 'rgba(250,0,0,1)');
    gradient.addColorStop(1, 'rgba(136,255,0,1)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent;
        const temp = item.querySelector('.forecast-temperatureValue')?.textContent;

        if (time && temp) {
            times.push(time);
            temps.push(Number(temp)); // Ensure temps are numbers
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing.');
        return;
    }

    // Use Chart with uppercase C
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true,
                    tension: 0.3, // smooth line
                },
            ],
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
});
