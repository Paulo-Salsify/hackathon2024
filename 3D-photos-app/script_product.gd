extends Sprite3D

var max_rotation_y = 30
var direction = 1
var velocity = 2.0
var step_increase = 1

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if absi(rotation.y) < max_rotation_y:
		rotation.y += delta * velocity * direction
	else:
		direction = direction * -1 
	
	max_rotation_y += step_increase
