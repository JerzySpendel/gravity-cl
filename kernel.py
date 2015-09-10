program = """
float planets_distance(const float4* p1, const float4* p2){
    float square1 = (p1->s0-p2->s0)*(p1->s0-p2->s0);
    float square2 = (p1->s1-p2->s1)*(p1->s1-p2->s1);
    return sqrt(square1 + square2);
}
float2 distance_vector(const float4* p1, const float4* p2){
    return (float2)(p2->s0 - p1->s0, p2->s1 - p1->s1);
}
int planets_are_equal(const float4* p1, const float4* p2){
    if(p1->s0 == p2->s0 && p1->s1 == p2->s1){
        return 1;
    }
    return 0;
}
int planets_collide(const float4* p1, const float4* p2){
    if(planets_distance(p1, p2) < 0.2){
        return 1;
    }
    return 0;
}
float2 apply_external_gravity_field(float4* p, const float2* click){
    if(click->s0 == 0.0f && click->s1 == 0.0f){
        return (float2)(0.0f, 0.0f);
    }
    float4 grav_planet = (float4)(click->s0, click->s1, 0.0, 0.0);
    if(planets_collide(&grav_planet, p) == 1){
        return (float2)(0.0f, 0.0f);
    }
    float2 r_vector = distance_vector(p, click);
    float r = planets_distance(p, click);
    float force_module = 1000.0/(r*r*r);
    float2 force = (float2)(r_vector.s0*force_module, r_vector.s1*force_module);
    return force;
}
void check_wall(float4* p){
    if(p->s0 < 0){
        p->s2 *= -1;
        p->s0 = 0;
    }
    else if(p->s0 > 500){
        p->s2 *= -1;
        p->s0 = 500;
    }
    else if(p->s1 < 0){
        p->s3 *= -1;
        p->s1 = 0;
    }
    else if(p->s1 > 500){
        p->s3 *= -1;
        p->s1 = 500;
    }
}
__kernel void apply_dt(__global float4* planet, const unsigned int n, float2 click, float dt){
    float2 force = (float2)(0.0f, 0.0f);
    float4 this_planet = planet[get_global_id(0)];
    float2 temp = (float2)(0.0f, 0.0f);
    for(int i = 0; i < n; i++){
        const float4 iter_planet = planet[i];
        if(planets_are_equal(&iter_planet, &this_planet) == 1){
            continue;
        }
        if(planets_collide(&iter_planet, &this_planet) == 1){
            continue;
        }
        check_wall(&this_planet);
        float r = planets_distance(&this_planet, &iter_planet);
        float2 r_vector = (float2)(iter_planet.s0 - this_planet.s0, iter_planet.s1 - this_planet.s1);
        float force_module = 1.0/(r*r*r);
        force.s0 += r_vector.s0*force_module;
        force.s1 += r_vector.s1*force_module;
    }
    float2 external_force = apply_external_gravity_field(&this_planet, &click);
    this_planet.s2 += force.s0 + external_force.s0;
    this_planet.s3 += force.s1 + external_force.s1;
    this_planet.s0 += this_planet.s2*dt;
    this_planet.s1 += this_planet.s3*dt;
    barrier(CLK_GLOBAL_MEM_FENCE);
    planet[get_global_id(0)] = this_planet;
}
"""