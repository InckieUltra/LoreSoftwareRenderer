struct VertexInput {
	@location(0) position: vec3f,
    //                        ^ This was a 2
	@location(1) normal: vec3f,
	@location(2) color: vec3f,
};

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) color: vec3f,
	@location(1) normal: vec3f,
};

/**
 * A structure holding the value of our uniforms
 */
struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    color: vec4f,
    time: f32,
};

// Instead of the simple uTime variable, our uniform variable is a struct
@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;

fn vs_main_optionB(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	out.position = uMyUniforms.projectionMatrix * uMyUniforms.viewMatrix * uMyUniforms.modelMatrix * vec4f(in.position, 1.0);
	out.color = in.color;
	out.normal = (uMyUniforms.modelMatrix * vec4f(in.normal, 0.0)).xyz;
	return out;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	return vs_main_optionB(in);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
	let normal = normalize(in.normal);
	
    let lightDirection = normalize(vec3f(0.5, 0.5, 0.5));  // 单一光源方向（必须归一化）
	let cameraPosition = -uMyUniforms.viewMatrix[3].xyz; // 相机在世界空间的位置
	let viewDirection = normalize(cameraPosition - in.position.xyz); // 从片元到相机的方向
	let reflectDirection = reflect(-lightDirection, normal); // 镜面反射方向

	let ambient = 0.1; // 环境光因子

	// 漫反射
	let diffuse = max(dot(normal, lightDirection), 0.0);

	// 镜面反射（Phong 模型）
	let specularStrength = 0.5;
	let shininess = 32.0;
	let specular = pow(max(dot(viewDirection, reflectDirection), 0.0), shininess) * specularStrength;

	// 最终光照强度
	let shading = ambient + diffuse + specular;

    //let color = in.color * shading;
	let color = vec3f(0.5, 0.5, 0.5) * shading;
	// Gamma-correction
	let corrected_color = pow(color, vec3f(2.2));
	return vec4f(corrected_color, uMyUniforms.color.a);
}