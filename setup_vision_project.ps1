$project = "retail-shelf-vision-service"

# ----------------------------
# Define classes (EDIT if needed)
# ----------------------------
$classes = @(
    "good_display",
    "low_stock",
    "empty_shelf",
    "misplaced_product",
    "poor_visibility"
)

# ----------------------------
# Create base folders
# ----------------------------
$folders = @(
    "$project",
    "$project\app",
    "$project\pipelines",
    "$project\configs",
    "$project\models",
    "$project\logs",
    "$project\data",
    "$project\data\raw",
    "$project\data\raw\train",
    "$project\data\raw\val",
    "$project\data\raw\test",
    "$project\data\processed",
    "$project\assets",
    "$project\tests",
    "$project\.github",
    "$project\.github\workflows"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

# ----------------------------
# Create class folders
# ----------------------------
foreach ($split in @("train", "val", "test")) {
    foreach ($class in $classes) {
        $path = "$project\data\raw\$split\$class"
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

# ----------------------------
# Create files
# ----------------------------
$files = @(
    "$project\app\__init__.py",
    "$project\app\main.py",
    "$project\app\config.py",
    "$project\app\schemas.py",
    "$project\app\predictor.py",
    "$project\app\utils.py",

    "$project\pipelines\__init__.py",
    "$project\pipelines\train.py",
    "$project\pipelines\batch_infer.py",
    "$project\pipelines\preprocess.py",

    "$project\configs\training_config.yaml",

    "$project\models\metadata.json",
    "$project\models\.gitkeep",

    "$project\logs\.gitkeep",

    "$project\data\processed\.gitkeep",

    "$project\assets\.gitkeep",

    "$project\tests\__init__.py",
    "$project\tests\test_api.py",

    "$project\.github\workflows\ci.yml",

    "$project\.gitignore",
    "$project\Dockerfile",
    "$project\requirements.txt",
    "$project\README.md",
    "$project\sample_request.json"
)

foreach ($file in $files) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

Write-Host "✅ Project + class folder structure created successfully!"